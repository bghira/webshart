use super::{header, invalid, FetchRange, Interval, MAX_DECODED_BYTES};
use crate::Result;
use reqwest::header::HeaderMap;
use std::collections::{HashMap, HashSet};
use std::io::Read;

fn content_range(value: &str) -> Result<Interval> {
    let bounds = value
        .strip_prefix("bytes ")
        .and_then(|value| value.split_once('/'))
        .and_then(|(bounds, _)| bounds.split_once('-'))
        .ok_or_else(|| invalid("Invalid Xet Content-Range"))?;
    let start = bounds
        .0
        .parse()
        .map_err(|_| invalid("Invalid Xet Content-Range"))?;
    let end = bounds
        .1
        .parse()
        .map_err(|_| invalid("Invalid Xet Content-Range"))?;
    let range = Interval { start, end };
    range.byte_length()?;
    Ok(range)
}

pub(super) fn parts<'data, 'ranges>(
    body: &'data [u8],
    headers: &HeaderMap,
    ranges: &'ranges [FetchRange],
) -> Result<Vec<(&'ranges FetchRange, &'data [u8])>> {
    if ranges.len() == 1 {
        let actual = content_range(
            header(headers, "content-range").ok_or_else(|| invalid("Missing Xet Content-Range"))?,
        )?;
        let range = &ranges[0];
        if actual.start != range.bytes.start
            || actual.end != range.bytes.end
            || body.len() as u64 != actual.byte_length()?
        {
            return Err(invalid("Xet byte range mismatch"));
        }
        return Ok(vec![(range, body)]);
    }
    let content_type = header(headers, "content-type")
        .ok_or_else(|| invalid("Missing multipart Xet Content-Type"))?;
    let mut parameters = content_type.split(';');
    if !parameters
        .next()
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("multipart/byteranges"))
    {
        return Err(invalid("Expected multipart Xet byte ranges"));
    }
    let boundary = parameters
        .filter_map(|value| value.trim().split_once('='))
        .find(|(key, _)| key.eq_ignore_ascii_case("boundary"))
        .map(|(_, value)| value.trim().trim_matches('"'))
        .filter(|value| {
            !value.is_empty()
                && value.len() <= 70
                && value.bytes().all(|byte| byte.is_ascii_graphic())
        })
        .ok_or_else(|| invalid("Invalid Xet multipart boundary"))?;
    let marker = format!("--{boundary}");
    let first_boundary = body
        .windows(marker.len())
        .take(16384)
        .enumerate()
        .find(|(offset, value)| {
            *value == marker.as_bytes() && (*offset == 0 || body[..*offset].ends_with(b"\r\n"))
        })
        .map(|(offset, _)| offset)
        .ok_or_else(|| invalid("Missing Xet multipart boundary"))?;
    let mut remaining = &body[first_boundary..];
    let mut parts = Vec::new();
    let mut seen = HashSet::new();
    loop {
        remaining = remaining
            .strip_prefix(marker.as_bytes())
            .ok_or_else(|| invalid("Invalid Xet multipart delimiter"))?;
        if remaining == b"--" || remaining.starts_with(b"--\r\n") {
            break;
        }
        remaining = remaining
            .strip_prefix(b"\r\n")
            .ok_or_else(|| invalid("Invalid Xet multipart delimiter"))?;
        let header_end = remaining
            .windows(4)
            .take(16384)
            .position(|bytes| bytes == b"\r\n\r\n")
            .ok_or_else(|| invalid("Invalid Xet multipart headers"))?;
        let text = std::str::from_utf8(&remaining[..header_end])
            .map_err(|_| invalid("Invalid Xet multipart headers"))?;
        let actual = text
            .split("\r\n")
            .filter_map(|line| line.split_once(':'))
            .find(|(key, _)| key.eq_ignore_ascii_case("content-range"))
            .ok_or_else(|| invalid("Missing Xet multipart Content-Range"))?;
        let actual = content_range(actual.1.trim())?;
        let range = ranges
            .iter()
            .find(|range| range.bytes.start == actual.start && range.bytes.end == actual.end)
            .ok_or_else(|| invalid("Unexpected Xet multipart range"))?;
        if !seen.insert(actual.start) {
            return Err(invalid("Duplicate Xet multipart range"));
        }
        remaining = &remaining[header_end + 4..];
        let length = usize::try_from(actual.byte_length()?)
            .map_err(|_| invalid("Xet range size overflow"))?;
        let part = remaining
            .get(..length)
            .ok_or_else(|| invalid("Truncated Xet multipart range"))?;
        parts.push((range, part));
        remaining = remaining[length..]
            .strip_prefix(b"\r\n")
            .ok_or_else(|| invalid("Invalid Xet multipart delimiter"))?;
    }
    if parts.len() != ranges.len() {
        return Err(invalid("Missing Xet multipart ranges"));
    }
    Ok(parts)
}

fn uint24(bytes: &[u8]) -> usize {
    usize::from(bytes[0]) | (usize::from(bytes[1]) << 8) | (usize::from(bytes[2]) << 16)
}

pub(super) fn decode_chunks(
    mut payload: &[u8],
    range: &Interval,
    needed: &HashSet<u64>,
) -> Result<HashMap<u64, Vec<u8>>> {
    let mut chunks = HashMap::new();
    let mut decoded_bytes = 0usize;
    for index in range.start..range.end {
        let header = payload
            .get(..8)
            .ok_or_else(|| invalid("Truncated Xet chunk header"))?;
        let compressed_size = uint24(&header[1..4]);
        let compression = header[4];
        let size = uint24(&header[5..8]);
        if header[0] != 0
            || compression > 2
            || size == 0
            || size > 128 * 1024
            || compressed_size > 256 * 1024
        {
            return Err(invalid("Invalid Xet chunk header"));
        }
        let encoded = payload
            .get(8..8 + compressed_size)
            .ok_or_else(|| invalid("Truncated Xet chunk"))?;
        payload = &payload[8 + compressed_size..];
        if !needed.contains(&index) {
            continue;
        }
        decoded_bytes += size;
        if decoded_bytes > MAX_DECODED_BYTES as usize {
            return Err(invalid("Xet decoded data exceeded its size limit"));
        }
        let mut decoded = if compression == 0 {
            encoded.to_vec()
        } else {
            let mut decoded = Vec::with_capacity(size);
            lz4_flex::frame::FrameDecoder::new(encoded)
                .take(size as u64 + 1)
                .read_to_end(&mut decoded)
                .map_err(|_| invalid("Invalid Xet LZ4 frame"))?;
            decoded
        };
        if decoded.len() != size {
            return Err(invalid("Xet chunk decompressed length mismatch"));
        }
        if compression == 2 {
            let mut ungrouped = vec![0; size];
            let mut cursor = 0;
            for lane in 0..4 {
                for output in ungrouped.iter_mut().skip(lane).step_by(4) {
                    *output = decoded[cursor];
                    cursor += 1;
                }
            }
            decoded = ungrouped;
        }
        chunks.insert(index, decoded);
    }
    if !payload.is_empty() {
        return Err(invalid("Trailing data after Xet chunk range"));
    }
    Ok(chunks)
}
