//! Optional byte-exact JPEG recompression with the reference JPEG XL tools.
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn run(command: &mut Command, timeout: Duration) -> Result<(), ()> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| ())?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return if status.success() { Ok(()) } else { Err(()) },
            Ok(None) if start.elapsed() < timeout => std::thread::sleep(Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(());
            }
        }
    }
}

pub fn available() -> bool {
    ["cjxl", "djxl"]
        .iter()
        .all(|tool| run(Command::new(tool).arg("--version"), Duration::from_secs(5)).is_ok())
}

fn encode(data: &[u8], root: &Path) -> Result<Option<Vec<u8>>, ()> {
    let directory = tempfile::tempdir_in(root).map_err(|_| ())?;
    let input = directory.path().join("input.jpg");
    let output = directory.path().join("output.jxl");
    let reconstructed = directory.path().join("reconstructed.jpg");
    std::fs::write(&input, data).map_err(|_| ())?;
    run(
        Command::new("cjxl").arg(&input).arg(&output).args([
            "--lossless_jpeg=1",
            "--effort=3",
            "--num_threads=1",
            "--quiet",
        ]),
        Duration::from_secs(60),
    )?;
    if std::fs::metadata(&output).map_err(|_| ())?.len() >= data.len() as u64 {
        return Ok(None);
    }
    run(
        Command::new("djxl")
            .arg(&output)
            .arg(&reconstructed)
            .args(["--num_threads=1", "--quiet"]),
        Duration::from_secs(60),
    )?;
    if std::fs::metadata(&reconstructed).map_err(|_| ())?.len() != data.len() as u64
        || std::fs::read(&reconstructed).map_err(|_| ())? != data
    {
        return Err(());
    }
    std::fs::read(output).map(Some).map_err(|_| ())
}

pub fn recompress(data: Vec<u8>, root: &Path) -> (Vec<u8>, &'static str) {
    if !data.starts_with(&[0xff, 0xd8, 0xff]) {
        return (data, "unsupported_format");
    }
    match encode(&data, root) {
        Ok(Some(encoded)) => (encoded, "recompressed"),
        Ok(None) => (data, "not_smaller"),
        Err(()) => (data, "codec_error"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_other_formats_and_invalid_jpeg() {
        let root = tempfile::tempdir().unwrap();
        let png = include_bytes!("../tests/fixtures/url-image.png").to_vec();
        assert_eq!(
            recompress(png.clone(), root.path()),
            (png, "unsupported_format")
        );
        let invalid = vec![0xff, 0xd8, 0xff, 0];
        assert_eq!(
            recompress(invalid.clone(), root.path()),
            (invalid, "codec_error")
        );
    }

    #[test]
    fn jpeg_roundtrip_when_tools_installed() {
        if !available() {
            return;
        }
        let root = tempfile::tempdir().unwrap();
        let jpeg = include_bytes!("../tests/fixtures/recompression.jpg").to_vec();
        let (encoded, status) = recompress(jpeg.clone(), root.path());
        assert_eq!(status, "recompressed");
        assert!(encoded.len() < jpeg.len());
    }
}
