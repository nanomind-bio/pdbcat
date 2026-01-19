//! Pixel buffer for raster rendering
//!
//! Stores per-pixel color and depth for image and half-block outputs.

/// A pixel buffer with depth testing.
#[derive(Debug, Clone)]
pub struct PixelBuffer {
    width: usize,
    height: usize,
    colors: Vec<(u8, u8, u8, u8)>,
    depth: Vec<f32>,
}

impl PixelBuffer {
    /// Create a new pixel buffer with the given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        let count = width * height;
        Self {
            width,
            height,
            colors: vec![(0, 0, 0, 0); count],
            depth: vec![f32::NEG_INFINITY; count],
        }
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.colors.fill((0, 0, 0, 0));
        self.depth.fill(f32::NEG_INFINITY);
    }

    /// Resize the buffer if dimensions changed, otherwise just clear.
    /// Returns true if buffer was resized, false if only cleared.
    pub fn resize_or_clear(&mut self, width: usize, height: usize) -> bool {
        if self.width != width || self.height != height {
            let count = width * height;
            self.width = width;
            self.height = height;
            self.colors = vec![(0, 0, 0, 0); count];
            self.depth = vec![f32::NEG_INFINITY; count];
            true
        } else {
            self.clear();
            false
        }
    }

    /// Pixel width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Pixel height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get a pixel's RGBA value.
    pub fn get_pixel(&self, x: usize, y: usize) -> (u8, u8, u8, u8) {
        if x >= self.width || y >= self.height {
            return (0, 0, 0, 0);
        }
        self.colors[y * self.width + x]
    }

    /// Set a pixel with depth testing.
    pub fn set_pixel(&mut self, x: i32, y: i32, z: f32, color: (u8, u8, u8)) -> bool {
        if x < 0 || y < 0 {
            return false;
        }
        let x = x as usize;
        let y = y as usize;
        if x >= self.width || y >= self.height {
            return false;
        }

        let idx = y * self.width + x;
        if z <= self.depth[idx] {
            return false;
        }
        self.depth[idx] = z;
        self.colors[idx] = (color.0, color.1, color.2, 255);
        true
    }

    /// Draw a line between two points.
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

    /// Draw a filled circle (flat shading).
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
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    let z_offset = (1.0 - dist / radius).max(0.0) * 0.5;
                    self.set_pixel(x, y, cz + z_offset, color);
                }
            }
        }
    }

    /// Draw a shaded sphere using Blinn-Phong lighting model.
    /// Creates a 3D appearance with diffuse and specular highlights.
    pub fn draw_sphere_shaded(
        &mut self,
        cx: f32,
        cy: f32,
        cz: f32,
        radius: f32,
        base_color: (u8, u8, u8),
    ) {
        if radius < 0.5 {
            self.set_pixel(cx as i32, cy as i32, cz, base_color);
            return;
        }

        // Light direction (normalized) - from upper-left-front
        let light = (0.4_f32, -0.5_f32, 0.76_f32);
        // View direction (towards viewer)
        let view = (0.0_f32, 0.0_f32, 1.0_f32);
        // Halfway vector for Blinn-Phong
        let h_len = ((light.0 + view.0).powi(2) + (light.1 + view.1).powi(2) + (light.2 + view.2).powi(2)).sqrt();
        let half = (
            (light.0 + view.0) / h_len,
            (light.1 + view.1) / h_len,
            (light.2 + view.2) / h_len,
        );

        let r2 = radius * radius;
        let min_x = (cx - radius).floor() as i32;
        let max_x = (cx + radius).ceil() as i32;
        let min_y = (cy - radius).floor() as i32;
        let max_y = (cy + radius).ceil() as i32;

        // Ambient light factor
        let ambient = 0.15_f32;
        // Diffuse strength
        let diffuse_strength = 0.7_f32;
        // Specular strength and shininess
        let specular_strength = 0.4_f32;
        let shininess = 32.0_f32;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = (x as f32 + 0.5) - cx;
                let dy = (y as f32 + 0.5) - cy;
                let d2 = dx * dx + dy * dy;

                if d2 > r2 {
                    continue;
                }

                // Calculate z on sphere surface
                let dz = (r2 - d2).sqrt();
                let z = cz + dz;

                // Normal at this point (normalized)
                let inv_r = 1.0 / radius;
                let nx = dx * inv_r;
                let ny = dy * inv_r;
                let nz = dz * inv_r;

                // Diffuse component (Lambert)
                let n_dot_l = (nx * light.0 + ny * light.1 + nz * light.2).max(0.0);
                let diffuse = diffuse_strength * n_dot_l;

                // Specular component (Blinn-Phong)
                let n_dot_h = (nx * half.0 + ny * half.1 + nz * half.2).max(0.0);
                let specular = specular_strength * n_dot_h.powf(shininess);

                // Combine lighting
                let shade = (ambient + diffuse).min(1.0);
                let color = (
                    ((base_color.0 as f32 * shade + 255.0 * specular).min(255.0)) as u8,
                    ((base_color.1 as f32 * shade + 255.0 * specular).min(255.0)) as u8,
                    ((base_color.2 as f32 * shade + 255.0 * specular).min(255.0)) as u8,
                );

                self.set_pixel(x, y, z, color);
            }
        }
    }

    /// Draw a shaded cylinder (bond) between two points.
    /// Creates a 3D tube appearance with lighting.
    pub fn draw_cylinder_shaded(
        &mut self,
        x0: f32,
        y0: f32,
        z0: f32,
        x1: f32,
        y1: f32,
        z1: f32,
        radius: f32,
        color: (u8, u8, u8),
    ) {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let dz = z1 - z0;
        let length = (dx * dx + dy * dy + dz * dz).sqrt();

        if length < 0.1 {
            return;
        }

        // Number of steps along the cylinder
        let steps = (length * 2.0).max(2.0) as i32;

        // Light direction
        let light = (0.4_f32, -0.5_f32, 0.76_f32);
        let ambient = 0.2_f32;
        let diffuse_strength = 0.8_f32;

        // Perpendicular vectors to the cylinder axis
        let axis = (dx / length, dy / length, dz / length);
        // Find a perpendicular vector
        let perp1 = if axis.0.abs() < 0.9 {
            let len = (axis.1 * axis.1 + axis.2 * axis.2).sqrt();
            if len > 0.001 {
                (0.0, -axis.2 / len, axis.1 / len)
            } else {
                (0.0, 1.0, 0.0)
            }
        } else {
            let len = (axis.0 * axis.0 + axis.2 * axis.2).sqrt();
            if len > 0.001 {
                (-axis.2 / len, 0.0, axis.0 / len)
            } else {
                (1.0, 0.0, 0.0)
            }
        };
        // Second perpendicular (cross product)
        let perp2 = (
            axis.1 * perp1.2 - axis.2 * perp1.1,
            axis.2 * perp1.0 - axis.0 * perp1.2,
            axis.0 * perp1.1 - axis.1 * perp1.0,
        );

        // Draw cylinder as series of circles
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let cx = x0 + dx * t;
            let cy = y0 + dy * t;
            let cz = z0 + dz * t;

            // Draw points around the circumference
            let angle_steps = ((radius * 4.0).max(4.0) as i32).min(16);
            for a in 0..angle_steps {
                let angle = (a as f32 / angle_steps as f32) * std::f32::consts::TAU;
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                // Point on cylinder surface
                let nx = perp1.0 * cos_a + perp2.0 * sin_a;
                let ny = perp1.1 * cos_a + perp2.1 * sin_a;
                let nz = perp1.2 * cos_a + perp2.2 * sin_a;

                let px = cx + nx * radius;
                let py = cy + ny * radius;
                let pz = cz + nz * radius;

                // Lighting
                let n_dot_l = (nx * light.0 + ny * light.1 + nz * light.2).max(0.0);
                let shade = (ambient + diffuse_strength * n_dot_l).min(1.0);

                let shaded_color = (
                    (color.0 as f32 * shade) as u8,
                    (color.1 as f32 * shade) as u8,
                    (color.2 as f32 * shade) as u8,
                );

                self.set_pixel(px as i32, py as i32, pz, shaded_color);
            }
        }
    }

    /// Draw an anti-aliased line using subpixel rendering.
    pub fn draw_line_aa(
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
        let length = (dx * dx + dy * dy).sqrt();
        let steps = (length * 2.0).max(1.0) as i32;

        for i in 0..=steps {
            let t = if steps > 0 { i as f32 / steps as f32 } else { 0.0 };
            let x = x0 + dx * t;
            let y = y0 + dy * t;
            let z = z0 + dz * t;

            // Plot main pixel and adjacent pixels with fractional coverage
            let xi = x.floor() as i32;
            let yi = y.floor() as i32;
            let fx = x - x.floor();
            let fy = y - y.floor();

            // Main pixel
            self.set_pixel(xi, yi, z, color);

            // Anti-aliased neighbors (simplified coverage)
            if fx > 0.3 {
                let blend = ((1.0 - fx) * 0.5) as f32;
                let aa_color = blend_color(color, blend);
                self.set_pixel(xi + 1, yi, z, aa_color);
            }
            if fy > 0.3 {
                let blend = ((1.0 - fy) * 0.5) as f32;
                let aa_color = blend_color(color, blend);
                self.set_pixel(xi, yi + 1, z, aa_color);
            }
        }
    }
}

/// Blend a color towards darker for anti-aliasing coverage
fn blend_color(color: (u8, u8, u8), factor: f32) -> (u8, u8, u8) {
    (
        (color.0 as f32 * factor) as u8,
        (color.1 as f32 * factor) as u8,
        (color.2 as f32 * factor) as u8,
    )
}

/// Downsample a buffer by 2x using box filtering for anti-aliasing.
/// Returns a new buffer at half the dimensions.
pub fn downsample_2x(src: &PixelBuffer) -> PixelBuffer {
    let dst_width = src.width() / 2;
    let dst_height = src.height() / 2;

    if dst_width == 0 || dst_height == 0 {
        return PixelBuffer::new(1, 1);
    }

    let mut dst = PixelBuffer::new(dst_width, dst_height);

    for y in 0..dst_height {
        for x in 0..dst_width {
            let mut r_sum: u32 = 0;
            let mut g_sum: u32 = 0;
            let mut b_sum: u32 = 0;
            let mut a_sum: u32 = 0;
            let mut max_z = f32::NEG_INFINITY;

            // Sample 2x2 block
            for oy in 0..2 {
                for ox in 0..2 {
                    let sx = x * 2 + ox;
                    let sy = y * 2 + oy;
                    let (r, g, b, a) = src.get_pixel(sx, sy);
                    r_sum += r as u32;
                    g_sum += g as u32;
                    b_sum += b as u32;
                    a_sum += a as u32;

                    // Track max depth for depth buffer
                    let idx = sy * src.width() + sx;
                    if idx < src.depth.len() && src.depth[idx] > max_z {
                        max_z = src.depth[idx];
                    }
                }
            }

            // Average colors
            let avg_r = (r_sum / 4) as u8;
            let avg_g = (g_sum / 4) as u8;
            let avg_b = (b_sum / 4) as u8;
            let avg_a = (a_sum / 4) as u8;

            let idx = y * dst_width + x;
            dst.colors[idx] = (avg_r, avg_g, avg_b, avg_a);
            dst.depth[idx] = max_z;
        }
    }

    dst
}
