//! Braille character rendering utilities
//!
//! Provides depth cueing for shading effects and half-block rendering.

use crate::render::PixelBuffer;
use std::io::{self, Write};

/// Apply depth cueing to a color
/// Makes objects farther from the camera darker for depth perception
pub fn depth_cue(color: (u8, u8, u8), z: f32, z_near: f32, z_far: f32) -> (u8, u8, u8) {
    if z_near >= z_far {
        return color;
    }

    // Normalize depth to 0.0 (far) to 1.0 (near)
    let depth_factor = (z - z_far) / (z_near - z_far);
    // Softer depth cue: far objects retain 50% brightness (was 30%)
    let depth_factor = depth_factor.clamp(0.5, 1.0);

    (
        (color.0 as f32 * depth_factor) as u8,
        (color.1 as f32 * depth_factor) as u8,
        (color.2 as f32 * depth_factor) as u8,
    )
}

/// Render a pixel buffer to stdout using half-block characters
/// Each character represents 2 vertical pixels using ▀ (upper half) with fg/bg colors
pub fn render_half_block(buffer: &PixelBuffer, out: &mut impl Write) -> io::Result<()> {
    let width = buffer.width();
    let height = buffer.height();

    // Process 2 rows at a time
    for y in (0..height).step_by(2) {
        for x in 0..width {
            let top = buffer.get_pixel(x, y);
            let bottom = if y + 1 < height {
                buffer.get_pixel(x, y + 1)
            } else {
                (0, 0, 0, 0)
            };

            // Use upper half block (▀) with top color as foreground, bottom as background
            let (tr, tg, tb, ta) = top;
            let (br, bg, bb, ba) = bottom;

            if ta == 0 && ba == 0 {
                // Both transparent - just space
                write!(out, " ")?;
            } else if ta == 0 {
                // Top transparent, bottom solid - use lower half block
                write!(out, "\x1b[38;2;{};{};{}m▄\x1b[0m", br, bg, bb)?;
            } else if ba == 0 {
                // Top solid, bottom transparent - use upper half block
                write!(out, "\x1b[38;2;{};{};{}m▀\x1b[0m", tr, tg, tb)?;
            } else {
                // Both solid - upper half with fg=top, bg=bottom
                write!(out, "\x1b[38;2;{};{};{};48;2;{};{};{}m▀\x1b[0m", tr, tg, tb, br, bg, bb)?;
            }
        }
        writeln!(out)?;
    }

    Ok(())
}
