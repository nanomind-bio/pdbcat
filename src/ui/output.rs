//! Terminal output backends for images and half-block fallback.

use crate::render::PixelBuffer;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::env;
use std::io::{self, Write};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageProtocol {
    Kitty,
    ITerm2,
    Sixel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderBackend {
    HalfBlock,
    Image(ImageProtocol),
}

impl RenderBackend {
    pub fn label(self) -> &'static str {
        match self {
            RenderBackend::HalfBlock => "Half-Block",
            RenderBackend::Image(ImageProtocol::Kitty) => "Kitty",
            RenderBackend::Image(ImageProtocol::ITerm2) => "iTerm2",
            RenderBackend::Image(ImageProtocol::Sixel) => "Sixel",
        }
    }
}

pub fn detect_backend() -> RenderBackend {
    if let Ok(value) = env::var("PDBCAT_IMAGE") {
        match value.to_lowercase().as_str() {
            "kitty" => return RenderBackend::Image(ImageProtocol::Kitty),
            "iterm2" | "iterm" => return RenderBackend::Image(ImageProtocol::ITerm2),
            "sixel" => return RenderBackend::Image(ImageProtocol::Sixel),
            "half" | "half-block" | "none" => return RenderBackend::HalfBlock,
            _ => {}
        }
    }

    let term = env::var("TERM").unwrap_or_default().to_lowercase();
    let term_program = env::var("TERM_PROGRAM").unwrap_or_default().to_lowercase();
    let lc_terminal = env::var("LC_TERMINAL").unwrap_or_default().to_lowercase();

    if env::var("KITTY_WINDOW_ID").is_ok() || term.contains("kitty") {
        return RenderBackend::Image(ImageProtocol::Kitty);
    }

    if has_iterm_env(&term_program, &lc_terminal) {
        return RenderBackend::Image(ImageProtocol::ITerm2);
    }

    if term.contains("sixel") || term.contains("mlterm") {
        return RenderBackend::Image(ImageProtocol::Sixel);
    }

    RenderBackend::HalfBlock
}

pub fn render_image(
    protocol: ImageProtocol,
    buffer: &PixelBuffer,
    cell_width: u16,
    cell_height: u16,
    out: &mut impl Write,
) -> io::Result<()> {
    let data = match protocol {
        ImageProtocol::Kitty => render_kitty_sequence(buffer, cell_width, cell_height),
        ImageProtocol::ITerm2 => render_iterm2_sequence(buffer, cell_width, cell_height),
        ImageProtocol::Sixel => render_sixel_sequence(buffer),
    };
    write_sequence(out, &data)
}

fn render_kitty_sequence(buffer: &PixelBuffer, cell_width: u16, cell_height: u16) -> Vec<u8> {
    let data = rgba_bytes(buffer);
    let payload = base64_encode(&data);

    let mut seq = Vec::new();
    seq.extend_from_slice(b"\x1b[H");
    seq.extend_from_slice(
        format!(
            "\x1b_Gf=32,s={},v={},c={},r={},a=T,i=1;",
            buffer.width(),
            buffer.height(),
            cell_width,
            cell_height
        )
        .as_bytes(),
    );
    seq.extend_from_slice(payload.as_bytes());
    seq.extend_from_slice(b"\x1b\\");
    seq
}

fn render_iterm2_sequence(buffer: &PixelBuffer, cell_width: u16, cell_height: u16) -> Vec<u8> {
    let png = encode_png(buffer);
    let payload = base64_encode(&png);

    // Use cell-based sizing with explicit "cell" units
    // doNotMoveCursor=1 prevents cursor jump/scroll after image
    let mut seq = Vec::new();
    seq.extend_from_slice(b"\x1b[H");
    seq.extend_from_slice(
        format!(
            "\x1b]1337;File=inline=1;size={};width={}cell;height={}cell;doNotMoveCursor=1:",
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

fn render_sixel_sequence(buffer: &PixelBuffer) -> Vec<u8> {
    let width = buffer.width();
    let height = buffer.height();
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let mut indices: Vec<u16> = Vec::with_capacity(width * height);
    let mut used = vec![false; 217];

    for y in 0..height {
        for x in 0..width {
            let (r, g, b, a) = buffer.get_pixel(x, y);
            let idx = if a == 0 {
                0
            } else {
                sixel_index(r, g, b)
            };
            indices.push(idx);
            if idx > 0 {
                used[idx as usize] = true;
            }
        }
    }

    let mut seq: Vec<u8> = Vec::new();
    seq.extend_from_slice(b"\x1b[H");
    // Start sixel with raster attributes: "Pan;Pad;Ph;Pv where:
    // Pan/Pad = pixel aspect ratio (1:1), Ph = width, Pv = height
    // This tells the terminal the intended image dimensions
    seq.extend_from_slice(format!("\x1bPq\"1;1;{};{}", width, height).as_bytes());
    seq.extend_from_slice(b"#0;2;0;0;0");

    for idx in 1..=216 {
        if used[idx] {
            let (r, g, b) = sixel_color(idx as u16);
            seq.extend_from_slice(format!("#{};2;{};{};{}", idx, r, g, b).as_bytes());
        }
    }

    let bands = (height + 5) / 6;
    for band in 0..bands {
        let band_start = band * 6;
        let band_end = (band_start + 6).min(height);
        let mut band_used = vec![false; 217];
        let mut band_colors: Vec<u16> = Vec::new();

        for y in band_start..band_end {
            for x in 0..width {
                let idx = indices[y * width + x];
                if idx > 0 && !band_used[idx as usize] {
                    band_used[idx as usize] = true;
                    band_colors.push(idx);
                }
            }
        }

        if band_colors.is_empty() {
            if band + 1 < bands {
                seq.push(b'-');
            }
            continue;
        }

        for (c_idx, color) in band_colors.iter().enumerate() {
            seq.extend_from_slice(format!("#{}", color).as_bytes());
            for x in 0..width {
                let mut bits: u8 = 0;
                for bit in 0..6 {
                    let y = band_start + bit;
                    if y < height && indices[y * width + x] == *color {
                        bits |= 1 << bit;
                    }
                }
                seq.push(63 + bits);
            }
            if c_idx + 1 < band_colors.len() {
                seq.push(b'$');
            }
        }

        if band + 1 < bands {
            seq.push(b'-');
        }
    }

    seq.extend_from_slice(b"\x1b\\");
    seq
}

fn rgba_bytes(buffer: &PixelBuffer) -> Vec<u8> {
    let mut data = Vec::with_capacity(buffer.width() * buffer.height() * 4);
    for y in 0..buffer.height() {
        for x in 0..buffer.width() {
            let (r, g, b, a) = buffer.get_pixel(x, y);
            data.extend_from_slice(&[r, g, b, a]);
        }
    }
    data
}

fn sixel_index(r: u8, g: u8, b: u8) -> u16 {
    let r6 = (r as u16 * 5 + 127) / 255;
    let g6 = (g as u16 * 5 + 127) / 255;
    let b6 = (b as u16 * 5 + 127) / 255;
    1 + r6 * 36 + g6 * 6 + b6
}

fn sixel_color(idx: u16) -> (u8, u8, u8) {
    let idx = idx.saturating_sub(1);
    let r = idx / 36;
    let g = (idx / 6) % 6;
    let b = idx % 6;
    (
        (r * 20) as u8,
        (g * 20) as u8,
        (b * 20) as u8,
    )
}

fn encode_png(buffer: &PixelBuffer) -> Vec<u8> {
    let width = buffer.width() as u32;
    let height = buffer.height() as u32;
    let mut raw = Vec::with_capacity((buffer.width() * buffer.height() * 4) + buffer.height());

    for y in 0..buffer.height() {
        raw.push(0);
        for x in 0..buffer.width() {
            let (r, g, b, a) = buffer.get_pixel(x, y);
            raw.extend_from_slice(&[r, g, b, a]);
        }
    }

    let compressed = zlib_compress(&raw);
    let mut png = Vec::new();
    png.extend_from_slice(b"\x89PNG\r\n\x1a\n");

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(6); // color type RGBA
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
