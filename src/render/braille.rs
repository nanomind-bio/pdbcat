//! Braille character rendering
//!
//! Implements a pseudo-pixel buffer using braille unicode characters.
//! Each terminal cell contains a 2×4 dot pattern, giving 8 subpixels per cell.

use nalgebra::Vector2;

/// Braille character base (empty pattern)
const BRAILLE_BASE: char = '\u{2800}';

/// Bit patterns for each dot position in braille character
/// Layout:
/// ```text
/// 0 3
/// 1 4
/// 2 5
/// 6 7
/// ```
const DOT_BITS: [u8; 8] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80];

/// A buffer for rendering using braille characters
#[derive(Debug, Clone)]
pub struct BrailleBuffer {
    /// Width in terminal cells
    width: usize,
    /// Height in terminal cells
    height: usize,
    /// Dot pattern for each cell (8 bits per cell)
    patterns: Vec<u8>,
    /// Color for each cell (RGB)
    colors: Vec<(u8, u8, u8)>,
    /// Depth buffer (one per subpixel)
    depth: Vec<f32>,
}

impl BrailleBuffer {
    /// Create a new buffer with the given size in terminal cells
    pub fn new(width: usize, height: usize) -> Self {
        let cell_count = width * height;
        let subpixel_count = width * 2 * height * 4;

        Self {
            width,
            height,
            patterns: vec![0; cell_count],
            colors: vec![(255, 255, 255); cell_count],
            depth: vec![f32::NEG_INFINITY; subpixel_count],
        }
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.patterns.fill(0);
        self.colors.fill((255, 255, 255));
        self.depth.fill(f32::NEG_INFINITY);
    }

    /// Get the width in terminal cells
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get the height in terminal cells
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get the width in subpixels
    pub fn pixel_width(&self) -> usize {
        self.width * 2
    }

    /// Get the height in subpixels
    pub fn pixel_height(&self) -> usize {
        self.height * 4
    }

    /// Set a subpixel with depth testing
    ///
    /// `x` and `y` are in subpixel coordinates (width*2, height*4)
    /// Returns true if the pixel was set (passed depth test)
    pub fn set_pixel(&mut self, x: i32, y: i32, z: f32, color: (u8, u8, u8)) -> bool {
        // Bounds check
        if x < 0 || y < 0 {
            return false;
        }
        let x = x as usize;
        let y = y as usize;
        if x >= self.pixel_width() || y >= self.pixel_height() {
            return false;
        }

        // Depth test
        let depth_idx = y * self.pixel_width() + x;
        if z <= self.depth[depth_idx] {
            return false;
        }
        self.depth[depth_idx] = z;

        // Calculate cell position
        let cell_x = x / 2;
        let cell_y = y / 4;
        let cell_idx = cell_y * self.width + cell_x;

        // Calculate dot position within cell
        let dot_x = x % 2;
        let dot_y = y % 4;
        let dot_idx = if dot_y < 3 {
            dot_y + dot_x * 3
        } else {
            6 + dot_x
        };

        // Set the dot
        self.patterns[cell_idx] |= DOT_BITS[dot_idx];

        // Update color (use the color of the frontmost dot)
        // We blend colors by using the most recent set
        self.colors[cell_idx] = color;

        true
    }

    /// Draw a line between two points with depth interpolation
    pub fn draw_line(
        &mut self,
        x0: f32,
        y0: f32,
        z0: f32,
        x1: f32,
        y1: f32,
        z1: f32,
        color: (u8, u8, u8),
    ) {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let dz = z1 - z0;

        let steps = dx.abs().max(dy.abs()).max(1.0) as i32;

        for i in 0..=steps {
            let t = if steps > 0 { i as f32 / steps as f32 } else { 0.0 };
            let x = x0 + dx * t;
            let y = y0 + dy * t;
            let z = z0 + dz * t;
            self.set_pixel(x as i32, y as i32, z, color);
        }
    }

    /// Draw a filled circle (for atoms)
    pub fn draw_circle(
        &mut self,
        cx: f32,
        cy: f32,
        cz: f32,
        radius: f32,
        color: (u8, u8, u8),
    ) {
        let r = radius.max(0.5) as i32;
        let r_sq = (radius * radius) as i32;

        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r_sq {
                    let x = cx as i32 + dx;
                    let y = cy as i32 + dy;
                    // Slight depth variation for sphere effect (closer at center)
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    let z_offset = (1.0 - dist / radius).max(0.0) * 0.5;
                    self.set_pixel(x, y, cz + z_offset, color);
                }
            }
        }
    }

    /// Get the braille character for a cell
    pub fn get_char(&self, cell_x: usize, cell_y: usize) -> char {
        if cell_x >= self.width || cell_y >= self.height {
            return ' ';
        }
        let idx = cell_y * self.width + cell_x;
        let pattern = self.patterns[idx];
        if pattern == 0 {
            ' '
        } else {
            char::from_u32(BRAILLE_BASE as u32 + pattern as u32).unwrap_or(' ')
        }
    }

    /// Get the color for a cell
    pub fn get_color(&self, cell_x: usize, cell_y: usize) -> (u8, u8, u8) {
        if cell_x >= self.width || cell_y >= self.height {
            return (255, 255, 255);
        }
        let idx = cell_y * self.width + cell_x;
        self.colors[idx]
    }

    /// Check if a cell has any dots set
    pub fn is_cell_empty(&self, cell_x: usize, cell_y: usize) -> bool {
        if cell_x >= self.width || cell_y >= self.height {
            return true;
        }
        let idx = cell_y * self.width + cell_x;
        self.patterns[idx] == 0
    }
}

/// Apply depth cueing to a color
pub fn depth_cue(color: (u8, u8, u8), z: f32, z_near: f32, z_far: f32) -> (u8, u8, u8) {
    if z_near >= z_far {
        return color;
    }

    // Normalize depth to 0.0 (far) to 1.0 (near)
    let depth_factor = (z - z_far) / (z_near - z_far);
    let depth_factor = depth_factor.clamp(0.3, 1.0);

    (
        (color.0 as f32 * depth_factor) as u8,
        (color.1 as f32 * depth_factor) as u8,
        (color.2 as f32 * depth_factor) as u8,
    )
}
