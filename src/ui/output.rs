//! Terminal output backends for images and half-block fallback.

use crate::render::PixelBuffer;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::env;
use std::io::{self, Write};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderBackend {
    HalfBlock,
    ITerm2,
}

impl RenderBackend {
    pub fn label(self) -> &'static str {
        match self {
            RenderBackend::HalfBlock => "Half-Block",
            RenderBackend::ITerm2 => "iTerm2",
        }
    }

    /// Check if this backend supports inline images
    pub fn supports_images(self) -> bool {
        matches!(self, RenderBackend::ITerm2)
    }
}

pub fn detect_backend() -> RenderBackend {
    if let Ok(value) = env::var("PDBCAT_IMAGE") {
        match value.to_lowercase().as_str() {
            "iterm2" | "iterm" => return RenderBackend::ITerm2,
            "half" | "half-block" | "none" => return RenderBackend::HalfBlock,
            _ => {}
        }
    }

    let term_program = env::var("TERM_PROGRAM").unwrap_or_default().to_lowercase();
    let lc_terminal = env::var("LC_TERMINAL").unwrap_or_default().to_lowercase();

    if has_iterm_env(&term_program, &lc_terminal) {
        return RenderBackend::ITerm2;
    }

    RenderBackend::HalfBlock
}

pub fn render_iterm2_image(
    buffer: &PixelBuffer,
    cell_width: u16,
    cell_height: u16,
    out: &mut impl Write,
) -> io::Result<()> {
    let data = render_iterm2_sequence(buffer, cell_width, cell_height);
    write_sequence(out, &data)
}

fn render_iterm2_sequence(buffer: &PixelBuffer, cell_width: u16, cell_height: u16) -> Vec<u8> {
    // Use RGB PNG (no alpha) - 25% smaller than RGBA
    let png = encode_png_rgb(buffer);
    let payload = base64_encode(&png);

    // preserveAspectRatio=0 prevents scaling blur by fitting exact dimensions
    // doNotMoveCursor=1 prevents cursor jump/scroll after image
    let mut seq = Vec::new();
    seq.extend_from_slice(b"\x1b[H");
    seq.extend_from_slice(
        format!(
            "\x1b]1337;File=inline=1;size={};width={}cell;height={}cell;preserveAspectRatio=0;doNotMoveCursor=1:",
            png.len(),
            cell_width,
            cell_height
        )
        .as_bytes(),
    );
    seq.extend_from_slice(payload.as_bytes());
    seq.extend_from_slice(b"\x07");
    seq
}

/// Encode as RGB PNG (3 bytes per pixel)
pub fn encode_png_rgb(buffer: &PixelBuffer) -> Vec<u8> {
    let width = buffer.width() as u32;
    let height = buffer.height() as u32;
    let mut raw = Vec::with_capacity((buffer.width() * buffer.height() * 3) + buffer.height());

    for y in 0..buffer.height() {
        raw.push(0); // filter byte
        for x in 0..buffer.width() {
            let (r, g, b, _) = buffer.get_pixel(x, y);
            raw.extend_from_slice(&[r, g, b]);
        }
    }

    let compressed = zlib_compress(&raw);
    let mut png = Vec::new();
    png.extend_from_slice(b"\x89PNG\r\n\x1a\n");

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(2); // color type RGB (no alpha)
    ihdr.push(0); // compression
    ihdr.push(0); // filter
    ihdr.push(0); // interlace
    write_chunk(&mut png, b"IHDR", &ihdr);
    write_chunk(&mut png, b"IDAT", &compressed);
    write_chunk(&mut png, b"IEND", &[]);

    png
}

fn zlib_compress(data: &[u8]) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(data).expect("zlib compression failed");
    encoder.finish().expect("zlib finish failed")
}

fn write_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    let crc = crc32(chunk_type, data);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// Pre-computed CRC32 lookup table for PNG (polynomial 0xEDB88320)
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

fn crc32(chunk_type: &[u8; 4], data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in chunk_type.iter().chain(data.iter()) {
        crc = CRC32_TABLE[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    let mut i = 0;

    while i < data.len() {
        let b0 = data[i];
        let b1 = if i + 1 < data.len() { data[i + 1] } else { 0 };
        let b2 = if i + 2 < data.len() { data[i + 2] } else { 0 };
        let triple = ((b0 as u32) << 16) | ((b1 as u32) << 8) | (b2 as u32);

        let idx0 = ((triple >> 18) & 0x3F) as usize;
        let idx1 = ((triple >> 12) & 0x3F) as usize;
        let idx2 = ((triple >> 6) & 0x3F) as usize;
        let idx3 = (triple & 0x3F) as usize;

        out.push(TABLE[idx0] as char);
        out.push(TABLE[idx1] as char);

        if i + 1 < data.len() {
            out.push(TABLE[idx2] as char);
        } else {
            out.push('=');
        }

        if i + 2 < data.len() {
            out.push(TABLE[idx3] as char);
        } else {
            out.push('=');
        }

        i += 3;
    }

    out
}

fn in_tmux() -> bool {
    env::var_os("TMUX").is_some()
}

fn has_iterm_env(term_program: &str, lc_terminal: &str) -> bool {
    if env::var_os("ITERM_SESSION_ID").is_some() {
        return true;
    }

    if term_program.contains("iterm") || lc_terminal.contains("iterm") {
        return true;
    }

    env::var_os("ITERM_PROFILE").is_some()
        || env::var_os("ITERM_ENABLE_SHELL_INTEGRATION_WITH_TMUX").is_some()
        || env::var_os("ITERM_ORIG_PS1").is_some()
        || env::var_os("ITERM_PREV_PS1").is_some()
}

fn write_sequence(out: &mut impl Write, data: &[u8]) -> io::Result<()> {
    if data.is_empty() {
        return Ok(());
    }

    if in_tmux() {
        let mut wrapped = Vec::with_capacity(data.len() + 16);
        wrapped.extend_from_slice(b"\x1bPtmux;");
        for &byte in data {
            if byte == 0x1b {
                wrapped.push(0x1b);
                wrapped.push(0x1b);
            } else {
                wrapped.push(byte);
            }
        }
        wrapped.extend_from_slice(b"\x1b\\");
        out.write_all(&wrapped)?;
    } else {
        out.write_all(data)?;
    }

    out.flush()
}
