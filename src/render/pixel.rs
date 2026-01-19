//! Pixel buffer for raster rendering
//!
//! Stores per-pixel color and depth for image and half-block outputs.

use rayon::prelude::*;

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
        let diffuse_strength = 0.65_f32;
        // Specular strength and shininess
        let specular_strength = 0.35_f32;
        let shininess = 32.0_f32;
        // Rim lighting strength (Fresnel-like edge highlight)
        let rim_strength = 0.3_f32;
        let rim_power = 2.5_f32;

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

                // Rim lighting (Fresnel-like effect at edges where nz approaches 0)
                let rim = rim_strength * (1.0 - nz).powf(rim_power);

                // Combine lighting
                let shade = (ambient + diffuse).min(1.0);
                let color = (
                    ((base_color.0 as f32 * shade + 255.0 * (specular + rim)).min(255.0)) as u8,
                    ((base_color.1 as f32 * shade + 255.0 * (specular + rim)).min(255.0)) as u8,
                    ((base_color.2 as f32 * shade + 255.0 * (specular + rim)).min(255.0)) as u8,
                );

                self.set_pixel(x, y, z, color);
            }
        }
    }

    /// Draw a shaded cylinder (bond) between two points.
    /// Uses filled disk rendering for solid appearance without gaps.
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

        // Light direction (same as spheres for consistency)
        let light = (0.4_f32, -0.5_f32, 0.76_f32);
        // View direction
        let view = (0.0_f32, 0.0_f32, 1.0_f32);
        // Halfway vector for specular
        let h_len = ((light.0 + view.0).powi(2) + (light.1 + view.1).powi(2) + (light.2 + view.2).powi(2)).sqrt();
        let half = (
            (light.0 + view.0) / h_len,
            (light.1 + view.1) / h_len,
            (light.2 + view.2) / h_len,
        );

        let ambient = 0.15_f32;
        let diffuse_strength = 0.65_f32;
        let specular_strength = 0.25_f32;
        let shininess = 24.0_f32;
        let rim_strength = 0.2_f32;
        let rim_power = 2.0_f32;

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

        // Draw cylinder as series of filled disks
        let steps = (length * 1.5).max(2.0) as i32;
        let r2 = radius * radius;
        let r_int = radius.ceil() as i32;

        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let cx = x0 + dx * t;
            let cy = y0 + dy * t;
            let cz = z0 + dz * t;

            // Fill the disk at this position
            for local_y in -r_int..=r_int {
                for local_x in -r_int..=r_int {
                    let dist_sq = (local_x * local_x + local_y * local_y) as f32;
                    if dist_sq > r2 {
                        continue;
                    }

                    // Convert local disk coordinates to world offset
                    let lx = local_x as f32;
                    let ly = local_y as f32;

                    // Offset in world space using perpendicular vectors
                    let wx = perp1.0 * lx + perp2.0 * ly;
                    let wy = perp1.1 * lx + perp2.1 * ly;
                    let wz = perp1.2 * lx + perp2.2 * ly;

                    let px = cx + wx;
                    let py = cy + wy;
                    let pz = cz + wz;

                    // Normal at this point (points outward from axis)
                    let dist = dist_sq.sqrt().max(0.001);
                    let nx = wx / dist;
                    let ny = wy / dist;
                    let nz_local = wz / dist;

                    // For depth, add z contribution from the cylinder surface
                    let surface_z = if dist < radius {
                        (r2 - dist_sq).sqrt() * 0.5  // Slight bulge for 3D effect
                    } else {
                        0.0
                    };

                    // Diffuse lighting
                    let n_dot_l = (nx * light.0 + ny * light.1 + nz_local * light.2).max(0.0);
                    let diffuse = diffuse_strength * n_dot_l;

                    // Specular (Blinn-Phong)
                    let n_dot_h = (nx * half.0 + ny * half.1 + nz_local * half.2).max(0.0);
                    let specular = specular_strength * n_dot_h.powf(shininess);

                    // Rim lighting
                    let n_dot_v = nz_local.abs();
                    let rim = rim_strength * (1.0 - n_dot_v).powf(rim_power);

                    let shade = (ambient + diffuse).min(1.0);
                    let shaded_color = (
                        ((color.0 as f32 * shade + 255.0 * (specular + rim)).min(255.0)) as u8,
                        ((color.1 as f32 * shade + 255.0 * (specular + rim)).min(255.0)) as u8,
                        ((color.2 as f32 * shade + 255.0 * (specular + rim)).min(255.0)) as u8,
                    );

                    self.set_pixel(px as i32, py as i32, pz + surface_z, shaded_color);
                }
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
/// Uses parallel processing for improved performance.
pub fn downsample_2x(src: &PixelBuffer) -> PixelBuffer {
    let dst_width = src.width() / 2;
    let dst_height = src.height() / 2;

    if dst_width == 0 || dst_height == 0 {
        return PixelBuffer::new(1, 1);
    }

    let src_width = src.width();

    // Compute each row in parallel
    let results: Vec<(Vec<(u8, u8, u8, u8)>, Vec<f32>)> = (0..dst_height)
        .into_par_iter()
        .map(|y| {
            let mut row_colors = Vec::with_capacity(dst_width);
            let mut row_depths = Vec::with_capacity(dst_width);

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

                        let idx = sy * src_width + sx;
                        if idx < src.depth.len() && src.depth[idx] > max_z {
                            max_z = src.depth[idx];
                        }
                    }
                }

                row_colors.push((
                    (r_sum / 4) as u8,
                    (g_sum / 4) as u8,
                    (b_sum / 4) as u8,
                    (a_sum / 4) as u8,
                ));
                row_depths.push(max_z);
            }

            (row_colors, row_depths)
        })
        .collect();

    // Assemble the final buffer
    let mut dst = PixelBuffer::new(dst_width, dst_height);
    for (y, (row_colors, row_depths)) in results.into_iter().enumerate() {
        for (x, (color, depth)) in row_colors.into_iter().zip(row_depths).enumerate() {
            let idx = y * dst_width + x;
            dst.colors[idx] = color;
            dst.depth[idx] = depth;
        }
    }

    dst
}

/// Apply silhouette edge detection to darken edges where depth changes sharply.
/// This creates a ChimeraX-style outline effect that enhances depth perception.
/// Uses parallel processing for improved performance.
pub fn apply_silhouette_edges(buffer: &mut PixelBuffer, strength: f32, threshold: f32) {
    let width = buffer.width;
    let height = buffer.height;

    if width < 3 || height < 3 {
        return;
    }

    // Compute edge factors in parallel (one row at a time)
    let edge_factors: Vec<f32> = (0..height)
        .into_par_iter()
        .flat_map(|y| {
            let mut row = vec![0.0_f32; width];

            // Skip first and last rows (boundary)
            if y == 0 || y == height - 1 {
                return row;
            }

            for x in 1..width - 1 {
                let idx = y * width + x;

                // Skip transparent pixels
                if buffer.colors[idx].3 == 0 {
                    continue;
                }

                // Sample 3x3 neighborhood depths
                let d_tl = buffer.depth[(y - 1) * width + (x - 1)];
                let d_t  = buffer.depth[(y - 1) * width + x];
                let d_tr = buffer.depth[(y - 1) * width + (x + 1)];
                let d_l  = buffer.depth[y * width + (x - 1)];
                let d_r  = buffer.depth[y * width + (x + 1)];
                let d_bl = buffer.depth[(y + 1) * width + (x - 1)];
                let d_b  = buffer.depth[(y + 1) * width + x];
                let d_br = buffer.depth[(y + 1) * width + (x + 1)];

                // Sobel gradient (horizontal and vertical)
                let gx = (d_tr + 2.0 * d_r + d_br) - (d_tl + 2.0 * d_l + d_bl);
                let gy = (d_bl + 2.0 * d_b + d_br) - (d_tl + 2.0 * d_t + d_tr);

                // Gradient magnitude
                let gradient = (gx * gx + gy * gy).sqrt();

                // Also detect edges at object boundaries (large depth discontinuities)
                let max_neighbor = d_tl.max(d_t).max(d_tr).max(d_l).max(d_r).max(d_bl).max(d_b).max(d_br);
                let min_neighbor = d_tl.min(d_t).min(d_tr).min(d_l).min(d_r).min(d_bl).min(d_b).min(d_br);
                let depth_range = max_neighbor - min_neighbor;

                // Combine gradient and depth discontinuity
                let edge_strength = gradient.max(depth_range * 0.5);

                if edge_strength > threshold {
                    // Normalize edge strength above threshold
                    let factor = ((edge_strength - threshold) * strength).min(0.7);
                    row[x] = factor;
                }
            }

            row
        })
        .collect();

    // Apply darkening in parallel using chunks
    buffer.colors
        .par_iter_mut()
        .zip(edge_factors.par_iter())
        .for_each(|(color, &factor)| {
            if factor > 0.0 {
                let darken = 1.0 - factor;
                *color = (
                    (color.0 as f32 * darken) as u8,
                    (color.1 as f32 * darken) as u8,
                    (color.2 as f32 * darken) as u8,
                    color.3,
                );
            }
        });
}
