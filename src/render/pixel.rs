//! Pixel buffer for raster rendering
//!
//! Stores per-pixel color and depth for image and half-block outputs.

use rayon::prelude::*;

/// Convert sRGB color component (0-255) to linear color space (0.0-1.0)
/// Uses the exact sRGB transfer function
#[inline]
fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert linear color component (0.0-1.0) to sRGB (0-255)
/// Uses the exact sRGB transfer function
#[inline]
fn linear_to_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let v = if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (v * 255.0 + 0.5) as u8
}

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

    /// Draw a shaded sphere using Blinn-Phong lighting model with gamma-correct shading.
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

        // Convert base color to linear space for correct lighting math
        let base_linear = (
            srgb_to_linear(base_color.0),
            srgb_to_linear(base_color.1),
            srgb_to_linear(base_color.2),
        );

        // Key light direction (normalized) - from upper-left-front
        let light_len = (0.4_f32 * 0.4 + 0.5 * 0.5 + 0.76 * 0.76).sqrt();
        let light = (0.4_f32 / light_len, -0.5_f32 / light_len, 0.76_f32 / light_len);
        // Fill light from opposite side (softer, dimmer)
        let fill_light = (-0.3_f32, 0.2_f32, 0.5_f32);
        let fill_strength = 0.25_f32;
        // View direction (towards viewer)
        let view = (0.0_f32, 0.0_f32, 1.0_f32);
        // Halfway vector for Blinn-Phong (key light)
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

        // Lighting parameters
        let ambient = 0.12_f32;
        let diffuse_strength = 0.70_f32;
        let specular_strength = 0.40_f32;
        let shininess = 40.0_f32;
        let rim_strength = 0.25_f32;
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

                // Key light diffuse (Lambert)
                let n_dot_l = (nx * light.0 + ny * light.1 + nz * light.2).max(0.0);
                let diffuse = diffuse_strength * n_dot_l;

                // Fill light diffuse (softer shadows)
                let n_dot_fill = (nx * fill_light.0 + ny * fill_light.1 + nz * fill_light.2).max(0.0);
                let fill_diffuse = fill_strength * n_dot_fill;

                // Specular component (Blinn-Phong) - GATED by n_dot_l
                let n_dot_h = (nx * half.0 + ny * half.1 + nz * half.2).max(0.0);
                let specular = if n_dot_l > 0.0 {
                    specular_strength * n_dot_h.powf(shininess)
                } else {
                    0.0
                };

                // Rim lighting (Fresnel-like effect at edges where nz approaches 0)
                let rim = rim_strength * (1.0 - nz).powf(rim_power);

                // Combine lighting in linear space
                let shade = ambient + diffuse + fill_diffuse;
                let lit = (
                    base_linear.0 * shade + specular + rim * base_linear.0,
                    base_linear.1 * shade + specular + rim * base_linear.1,
                    base_linear.2 * shade + specular + rim * base_linear.2,
                );

                // Convert back to sRGB
                let color = (
                    linear_to_srgb(lit.0),
                    linear_to_srgb(lit.1),
                    linear_to_srgb(lit.2),
                );

                self.set_pixel(x, y, z, color);
            }
        }
    }

    /// Draw a shaded cylinder (bond) between two points with gamma-correct shading.
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

        // Convert base color to linear space
        let base_linear = (
            srgb_to_linear(color.0),
            srgb_to_linear(color.1),
            srgb_to_linear(color.2),
        );

        // Key light direction (normalized) - same as spheres for consistency
        let light_len = (0.4_f32 * 0.4 + 0.5 * 0.5 + 0.76 * 0.76).sqrt();
        let light = (0.4_f32 / light_len, -0.5_f32 / light_len, 0.76_f32 / light_len);
        // Fill light from opposite side
        let fill_light = (-0.3_f32, 0.2_f32, 0.5_f32);
        let fill_strength = 0.25_f32;
        // View direction
        let view = (0.0_f32, 0.0_f32, 1.0_f32);
        // Halfway vector for specular
        let h_len = ((light.0 + view.0).powi(2) + (light.1 + view.1).powi(2) + (light.2 + view.2).powi(2)).sqrt();
        let half = (
            (light.0 + view.0) / h_len,
            (light.1 + view.1) / h_len,
            (light.2 + view.2) / h_len,
        );

        // Lighting parameters
        let ambient = 0.12_f32;
        let diffuse_strength = 0.70_f32;
        let specular_strength = 0.30_f32;
        let shininess = 32.0_f32;
        let rim_strength = 0.15_f32;
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

                    // Key light diffuse
                    let n_dot_l = (nx * light.0 + ny * light.1 + nz_local * light.2).max(0.0);
                    let diffuse = diffuse_strength * n_dot_l;

                    // Fill light diffuse
                    let n_dot_fill = (nx * fill_light.0 + ny * fill_light.1 + nz_local * fill_light.2).max(0.0);
                    let fill_diffuse = fill_strength * n_dot_fill;

                    // Specular (Blinn-Phong) - GATED by n_dot_l
                    let n_dot_h = (nx * half.0 + ny * half.1 + nz_local * half.2).max(0.0);
                    let specular = if n_dot_l > 0.0 {
                        specular_strength * n_dot_h.powf(shininess)
                    } else {
                        0.0
                    };

                    // Rim lighting
                    let n_dot_v = nz_local.abs();
                    let rim = rim_strength * (1.0 - n_dot_v).powf(rim_power);

                    // Combine lighting in linear space
                    let shade = ambient + diffuse + fill_diffuse;
                    let lit = (
                        base_linear.0 * shade + specular + rim * base_linear.0,
                        base_linear.1 * shade + specular + rim * base_linear.1,
                        base_linear.2 * shade + specular + rim * base_linear.2,
                    );

                    // Convert back to sRGB
                    let shaded_color = (
                        linear_to_srgb(lit.0),
                        linear_to_srgb(lit.1),
                        linear_to_srgb(lit.2),
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
/// IMPORTANT: Blends in linear color space for correct anti-aliasing.
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
                // Linear-space accumulation with alpha weighting
                let mut r_lin: f32 = 0.0;
                let mut g_lin: f32 = 0.0;
                let mut b_lin: f32 = 0.0;
                let mut a_sum: f32 = 0.0;
                let mut max_z = f32::NEG_INFINITY;

                // Sample 2x2 block
                for oy in 0..2 {
                    for ox in 0..2 {
                        let sx = x * 2 + ox;
                        let sy = y * 2 + oy;
                        let (r, g, b, a) = src.get_pixel(sx, sy);

                        // Convert to linear space and weight by alpha
                        let alpha = a as f32 / 255.0;
                        r_lin += srgb_to_linear(r) * alpha;
                        g_lin += srgb_to_linear(g) * alpha;
                        b_lin += srgb_to_linear(b) * alpha;
                        a_sum += alpha;

                        let idx = sy * src_width + sx;
                        if idx < src.depth.len() && src.depth[idx] > max_z {
                            max_z = src.depth[idx];
                        }
                    }
                }

                // Normalize and convert back to sRGB
                let (r, g, b, a) = if a_sum > 0.0 {
                    (
                        linear_to_srgb(r_lin / a_sum),
                        linear_to_srgb(g_lin / a_sum),
                        linear_to_srgb(b_lin / a_sum),
                        (a_sum * 255.0 / 4.0) as u8,
                    )
                } else {
                    (0, 0, 0, 0)
                };

                row_colors.push((r, g, b, a));
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
/// Now uses normalized depth thresholds relative to scene depth range.
pub fn apply_silhouette_edges(buffer: &mut PixelBuffer, strength: f32, _threshold: f32) {
    let width = buffer.width;
    let height = buffer.height;

    if width < 3 || height < 3 {
        return;
    }

    // Find the depth range of visible pixels (excluding background)
    let (min_depth, max_depth) = buffer.depth
        .par_iter()
        .filter(|&&d| d > f32::NEG_INFINITY)
        .fold(
            || (f32::INFINITY, f32::NEG_INFINITY),
            |(min_d, max_d), &d| (min_d.min(d), max_d.max(d)),
        )
        .reduce(
            || (f32::INFINITY, f32::NEG_INFINITY),
            |(a_min, a_max), (b_min, b_max)| (a_min.min(b_min), a_max.max(b_max)),
        );

    let depth_range = (max_depth - min_depth).max(1.0);
    // Normalized threshold: detect edges at ~2% of depth range
    let norm_threshold = depth_range * 0.02;

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

                let center_depth = buffer.depth[idx];

                // Skip if center is background (shouldn't happen since we check alpha)
                if center_depth <= f32::NEG_INFINITY {
                    continue;
                }

                // Sample 3x3 neighborhood depths, treating background as center depth
                // to avoid halos at object boundaries
                let sample = |dx: i32, dy: i32| -> f32 {
                    let nx = (x as i32 + dx) as usize;
                    let ny = (y as i32 + dy) as usize;
                    let n_idx = ny * width + nx;
                    let n_alpha = buffer.colors[n_idx].3;
                    let n_depth = buffer.depth[n_idx];
                    // If neighbor is background/transparent, use center depth
                    if n_alpha == 0 || n_depth <= f32::NEG_INFINITY {
                        center_depth
                    } else {
                        n_depth
                    }
                };

                let d_tl = sample(-1, -1);
                let d_t  = sample(0, -1);
                let d_tr = sample(1, -1);
                let d_l  = sample(-1, 0);
                let d_r  = sample(1, 0);
                let d_bl = sample(-1, 1);
                let d_b  = sample(0, 1);
                let d_br = sample(1, 1);

                // Sobel gradient (horizontal and vertical)
                let gx = (d_tr + 2.0 * d_r + d_br) - (d_tl + 2.0 * d_l + d_bl);
                let gy = (d_bl + 2.0 * d_b + d_br) - (d_tl + 2.0 * d_t + d_tr);

                // Gradient magnitude normalized to depth range
                let gradient = (gx * gx + gy * gy).sqrt();

                // Also detect edges between objects at different depths
                let max_neighbor = d_tl.max(d_t).max(d_tr).max(d_l).max(d_r).max(d_bl).max(d_b).max(d_br);
                let min_neighbor = d_tl.min(d_t).min(d_tr).min(d_l).min(d_r).min(d_bl).min(d_b).min(d_br);
                let local_range = max_neighbor - min_neighbor;

                // Combine gradient and depth discontinuity
                let edge_strength = gradient.max(local_range * 0.5);

                if edge_strength > norm_threshold {
                    // Normalize edge strength to 0-1 range, then apply strength
                    let normalized = ((edge_strength - norm_threshold) / depth_range).min(1.0);
                    let factor = (normalized * strength * 2.0).min(0.7);
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

/// SSAO sample kernel - Poisson disk offsets for ambient occlusion sampling
const SSAO_KERNEL: [(f32, f32); 12] = [
    (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
    (0.707, 0.707), (-0.707, 0.707), (0.707, -0.707), (-0.707, -0.707),
    (0.5, 0.866), (-0.5, 0.866), (0.5, -0.866), (-0.5, -0.866),
];

/// Apply Screen Space Ambient Occlusion for enhanced depth perception.
/// Creates soft shadows in crevices and contact areas between objects.
/// Uses parallel processing for performance.
pub fn apply_ssao(buffer: &mut PixelBuffer, radius: f32, strength: f32) {
    let width = buffer.width;
    let height = buffer.height;

    if width < 5 || height < 5 {
        return;
    }

    // Find depth range of visible pixels for normalization
    let (min_depth, max_depth) = buffer.depth
        .par_iter()
        .filter(|&&d| d > f32::NEG_INFINITY)
        .fold(
            || (f32::INFINITY, f32::NEG_INFINITY),
            |(min_d, max_d), &d| (min_d.min(d), max_d.max(d)),
        )
        .reduce(
            || (f32::INFINITY, f32::NEG_INFINITY),
            |(a_min, a_max), (b_min, b_max)| (a_min.min(b_min), a_max.max(b_max)),
        );

    let depth_range = (max_depth - min_depth).max(1.0);
    // Scale sample radius based on image size (larger images need larger samples)
    let sample_radius = (radius * (width.min(height) as f32 / 200.0).max(1.0)).max(2.0);
    // Depth threshold for occlusion (relative to depth range)
    let depth_threshold = depth_range * 0.01;

    // Compute occlusion factors in parallel
    let occlusion: Vec<f32> = (0..height)
        .into_par_iter()
        .flat_map(|y| {
            let mut row = vec![0.0_f32; width];

            for x in 0..width {
                let idx = y * width + x;

                // Skip background pixels
                if buffer.colors[idx].3 == 0 {
                    continue;
                }

                let center_depth = buffer.depth[idx];
                if center_depth <= f32::NEG_INFINITY {
                    continue;
                }

                // Sample surrounding depths using kernel
                let mut occluded_count = 0;
                let mut valid_samples = 0;

                for &(kx, ky) in &SSAO_KERNEL {
                    // Sample at multiple radii for better quality
                    for scale in &[0.5_f32, 1.0, 1.5] {
                        let sx = x as i32 + (kx * sample_radius * scale) as i32;
                        let sy = y as i32 + (ky * sample_radius * scale) as i32;

                        // Bounds check
                        if sx < 0 || sy < 0 || sx >= width as i32 || sy >= height as i32 {
                            continue;
                        }

                        let s_idx = sy as usize * width + sx as usize;
                        let sample_alpha = buffer.colors[s_idx].3;

                        // Skip background samples
                        if sample_alpha == 0 {
                            continue;
                        }

                        let sample_depth = buffer.depth[s_idx];
                        if sample_depth <= f32::NEG_INFINITY {
                            continue;
                        }

                        valid_samples += 1;

                        // If sample is significantly in front of center, center is occluded
                        if sample_depth > center_depth + depth_threshold {
                            // Weight by how much it occludes (closer occluders = stronger)
                            let occlusion_strength = ((sample_depth - center_depth) / depth_range).min(1.0);
                            if occlusion_strength > 0.02 {
                                occluded_count += 1;
                            }
                        }
                    }
                }

                // Compute occlusion factor
                if valid_samples > 0 {
                    let occ_ratio = occluded_count as f32 / valid_samples as f32;
                    // Apply smooth falloff
                    row[x] = (occ_ratio * strength).min(0.5);
                }
            }

            row
        })
        .collect();

    // Apply occlusion darkening in linear space
    buffer.colors
        .par_iter_mut()
        .zip(occlusion.par_iter())
        .for_each(|(color, &occ)| {
            if occ > 0.0 {
                // Darken in linear space for physically correct result
                let r_lin = srgb_to_linear(color.0);
                let g_lin = srgb_to_linear(color.1);
                let b_lin = srgb_to_linear(color.2);

                let darken = 1.0 - occ;
                *color = (
                    linear_to_srgb(r_lin * darken),
                    linear_to_srgb(g_lin * darken),
                    linear_to_srgb(b_lin * darken),
                    color.3,
                );
            }
        });
}

/// ACES Filmic Tone Mapping curve
/// Attempt to simulate the Academy Color Encoding System response
#[inline]
fn aces_tonemap(x: f32) -> f32 {
    // Simplified ACES approximation (Krzysztof Narkowicz)
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

/// Apply ACES filmic tone mapping to the buffer for professional color reproduction.
/// This maps HDR linear values to a pleasing display range with proper highlight rolloff.
/// Uses parallel processing for performance.
pub fn apply_tone_mapping(buffer: &mut PixelBuffer, exposure: f32) {
    buffer.colors
        .par_iter_mut()
        .for_each(|color| {
            if color.3 == 0 {
                return;
            }

            // Convert to linear space
            let r_lin = srgb_to_linear(color.0) * exposure;
            let g_lin = srgb_to_linear(color.1) * exposure;
            let b_lin = srgb_to_linear(color.2) * exposure;

            // Apply ACES tone mapping
            let r_tm = aces_tonemap(r_lin);
            let g_tm = aces_tonemap(g_lin);
            let b_tm = aces_tonemap(b_lin);

            // Convert back to sRGB
            *color = (
                linear_to_srgb(r_tm),
                linear_to_srgb(g_tm),
                linear_to_srgb(b_tm),
                color.3,
            );
        });
}
