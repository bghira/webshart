use std::mem::MaybeUninit;
use std::ptr::{null, NonNull};

use jxl_sys::*;
use turbojpeg::{Colorspace, Decompressor, Image, PixelFormat};

pub struct Pixels {
    pub width: u32,
    pub height: u32,
    pub mode: &'static str,
    pub data: Vec<u8>,
    pub exif: Option<Vec<u8>>,
    pub icc_profile: Vec<u8>,
}

struct Decoder(NonNull<JxlDecoder>);

impl Drop for Decoder {
    fn drop(&mut self) {
        // The pointer is solely owned by this decoder and came from JxlDecoderCreate.
        unsafe { JxlDecoderDestroy(self.0.as_ptr()) };
    }
}

fn success(status: JxlDecoderStatus) -> Result<(), String> {
    if status == JxlDecoderStatus::JXL_DEC_SUCCESS {
        Ok(())
    } else {
        Err(format!("JPEG XL decoder returned status {status:?}"))
    }
}

fn checked_pixels(width: usize, height: usize, max_pixels: usize) -> Result<usize, String> {
    width
        .checked_mul(height)
        .filter(|size| *size > 0 && *size <= max_pixels)
        .ok_or_else(|| format!("JPEG XL image exceeds the {max_pixels}-pixel limit"))
}

fn decode_jpeg(
    data: &[u8],
    max_pixels: usize,
    exif: Option<Vec<u8>>,
    icc_profile: Vec<u8>,
) -> Result<Pixels, String> {
    let mut decoder = Decompressor::new().map_err(|error| error.to_string())?;
    let header = decoder
        .read_header(data)
        .map_err(|error| error.to_string())?;
    let size = checked_pixels(header.width, header.height, max_pixels)?;
    let (mode, format) = match header.colorspace {
        Colorspace::Gray => ("L", PixelFormat::GRAY),
        Colorspace::CMYK | Colorspace::YCCK => ("CMYK", PixelFormat::CMYK),
        _ => ("RGB", PixelFormat::RGB),
    };
    let pitch = header.width * format.size();
    let mut pixels = vec![0; size * format.size()];
    decoder
        .decompress(
            data,
            Image {
                pixels: &mut pixels,
                width: header.width,
                height: header.height,
                pitch,
                format,
            },
        )
        .map_err(|error| error.to_string())?;
    Ok(Pixels {
        width: header.width as u32,
        height: header.height as u32,
        mode,
        data: pixels,
        exif,
        icc_profile,
    })
}

fn exif_payload(mut data: Vec<u8>) -> Option<Vec<u8>> {
    let offset = u32::from_be_bytes(data.get(..4)?.try_into().ok()?) as usize;
    let start = 4usize.checked_add(offset)?;
    data.get(start..)?;
    data.drain(..start);
    Some(data)
}

struct Output {
    buffer: Vec<u8>,
    written: usize,
}

impl Output {
    fn new() -> Self {
        Self {
            buffer: Vec::new(),
            written: 0,
        }
    }

    fn grow(&mut self, limit: usize) -> Result<&mut [u8], String> {
        let length = self.buffer.len().saturating_mul(2).max(4096).min(limit);
        if length <= self.written {
            return Err("JPEG XL metadata or JPEG output exceeds the decode limit".into());
        }
        self.buffer.resize(length, 0);
        Ok(&mut self.buffer[self.written..])
    }

    fn release(&mut self, unused: usize) {
        self.written = self.buffer.len() - unused;
    }

    fn finish(mut self) -> Vec<u8> {
        self.buffer.truncate(self.written);
        self.buffer
    }
}

pub fn decode(
    payload: &[u8],
    max_pixels: usize,
    allocation_limit: usize,
) -> Result<Pixels, String> {
    let mut jpeg = Output::new();
    let mut box_data = Output::new();
    let mut pixels = Vec::new();
    let mut icc_profile = Vec::new();
    let mut info: Option<JxlBasicInfo> = None;
    let mut jpeg_started = false;
    let mut reading_exif = false;
    let mut first_frame_ready = false;
    let mut exif = None;

    // libjxl borrows the input and output buffers until release or destruction. All
    // buffers outlive the decoder. Streaming buffers grow only after being released.
    unsafe {
        let decoder =
            Decoder(NonNull::new(JxlDecoderCreate(null())).ok_or("cannot create JPEG XL decoder")?);
        let dec = decoder.0.as_ptr();
        let events = JxlDecoderStatus::JXL_DEC_BASIC_INFO as i32
            | JxlDecoderStatus::JXL_DEC_COLOR_ENCODING as i32
            | JxlDecoderStatus::JXL_DEC_JPEG_RECONSTRUCTION as i32
            | JxlDecoderStatus::JXL_DEC_FULL_IMAGE as i32
            | JxlDecoderStatus::JXL_DEC_BOX as i32;
        success(JxlDecoderSubscribeEvents(dec, events))?;
        success(JxlDecoderSetDecompressBoxes(dec, 1))?;
        success(JxlDecoderSetInput(dec, payload.as_ptr(), payload.len()))?;
        JxlDecoderCloseInput(dec);
        loop {
            match JxlDecoderProcessInput(dec) {
                JxlDecoderStatus::JXL_DEC_BASIC_INFO => {
                    let mut out = MaybeUninit::uninit();
                    success(JxlDecoderGetBasicInfo(dec, out.as_mut_ptr()))?;
                    let header: JxlBasicInfo = out.assume_init();
                    checked_pixels(header.xsize as usize, header.ysize as usize, max_pixels)?;
                    info = Some(header);
                }
                JxlDecoderStatus::JXL_DEC_COLOR_ENCODING => {
                    let target = JxlColorProfileTarget::JXL_COLOR_PROFILE_TARGET_DATA;
                    let mut size = 0;
                    if JxlDecoderGetICCProfileSize(dec, target, &mut size)
                        == JxlDecoderStatus::JXL_DEC_SUCCESS
                    {
                        if size > allocation_limit {
                            return Err("oversized JPEG XL ICC profile".into());
                        }
                        icc_profile.resize(size, 0);
                        success(JxlDecoderGetColorAsICCProfile(
                            dec,
                            target,
                            icc_profile.as_mut_ptr(),
                            size,
                        ))?;
                    }
                }
                JxlDecoderStatus::JXL_DEC_JPEG_RECONSTRUCTION
                | JxlDecoderStatus::JXL_DEC_JPEG_NEED_MORE_OUTPUT => {
                    if jpeg_started {
                        jpeg.release(JxlDecoderReleaseJPEGBuffer(dec));
                    }
                    let buffer = jpeg.grow(allocation_limit)?;
                    success(JxlDecoderSetJPEGBuffer(
                        dec,
                        buffer.as_mut_ptr(),
                        buffer.len(),
                    ))?;
                    jpeg_started = true;
                }
                JxlDecoderStatus::JXL_DEC_NEED_IMAGE_OUT_BUFFER => {
                    let header = info
                        .as_ref()
                        .ok_or("JPEG XL frame lacks basic information")?;
                    let format = JxlPixelFormat {
                        num_channels: header.num_color_channels + u32::from(header.alpha_bits != 0),
                        data_type: JxlDataType::JXL_TYPE_UINT8,
                        endianness: JxlEndianness::JXL_NATIVE_ENDIAN,
                        align: 0,
                    };
                    let mut size = 0;
                    success(JxlDecoderImageOutBufferSize(dec, &format, &mut size))?;
                    if size > allocation_limit {
                        return Err("oversized JPEG XL image buffer".into());
                    }
                    pixels.resize(size, 0);
                    success(JxlDecoderSetImageOutBuffer(
                        dec,
                        &format,
                        pixels.as_mut_ptr().cast(),
                        pixels.len(),
                    ))?;
                }
                JxlDecoderStatus::JXL_DEC_FULL_IMAGE => {
                    first_frame_ready = true;
                    JxlDecoderSkipFrames(dec, usize::MAX);
                }
                JxlDecoderStatus::JXL_DEC_BOX => {
                    if reading_exif {
                        box_data.release(JxlDecoderReleaseBoxBuffer(dec));
                        exif =
                            exif_payload(std::mem::replace(&mut box_data, Output::new()).finish());
                        reading_exif = false;
                    }
                    let mut kind = [0u8; 4];
                    success(JxlDecoderGetBoxType(dec, kind.as_mut_ptr().cast(), 1))?;
                    if &kind == b"Exif" && exif.is_none() {
                        let buffer = box_data.grow(allocation_limit)?;
                        success(JxlDecoderSetBoxBuffer(
                            dec,
                            buffer.as_mut_ptr(),
                            buffer.len(),
                        ))?;
                        reading_exif = true;
                    }
                }
                JxlDecoderStatus::JXL_DEC_BOX_NEED_MORE_OUTPUT if reading_exif => {
                    box_data.release(JxlDecoderReleaseBoxBuffer(dec));
                    let buffer = box_data.grow(allocation_limit)?;
                    success(JxlDecoderSetBoxBuffer(
                        dec,
                        buffer.as_mut_ptr(),
                        buffer.len(),
                    ))?;
                }
                JxlDecoderStatus::JXL_DEC_SUCCESS => {
                    if reading_exif {
                        box_data.release(JxlDecoderReleaseBoxBuffer(dec));
                        exif = exif_payload(box_data.finish());
                    }
                    if jpeg_started {
                        jpeg.release(JxlDecoderReleaseJPEGBuffer(dec));
                    }
                    break;
                }
                JxlDecoderStatus::JXL_DEC_NEED_MORE_INPUT => {
                    return Err("incomplete JPEG XL image".into())
                }
                status => return Err(format!("invalid JPEG XL image (decoder status {status:?})")),
            }
        }
    }

    if jpeg_started {
        return decode_jpeg(&jpeg.finish(), max_pixels, exif, icc_profile);
    }
    if !first_frame_ready || pixels.is_empty() {
        return Err("JPEG XL image has no decoded frame".into());
    }
    let header = info.ok_or("JPEG XL frame lacks basic information")?;
    let mode = match (header.num_color_channels, header.alpha_bits != 0) {
        (1, false) => "L",
        (1, true) => "LA",
        (3, false) => "RGB",
        (3, true) => "RGBA",
        _ => return Err("unsupported JPEG XL channel layout".into()),
    };
    let expected =
        checked_pixels(header.xsize as usize, header.ysize as usize, max_pixels)? * mode.len();
    if expected != pixels.len() {
        return Err("unexpected JPEG XL frame size".into());
    }
    Ok(Pixels {
        width: header.xsize,
        height: header.ysize,
        mode,
        data: pixels,
        exif,
        icc_profile,
    })
}
