//! Pixel buffer for raster rendering
//!
//! Stores per-pixel color and depth for image and half-block outputs.

use rayon::prelude::*;

/// Precomputed sRGB to linear lookup table (256 entries)
/// Eliminates expensive pow() calls in hot paths
const SRGB_TO_LINEAR_LUT: [f32; 256] = {
    let mut lut = [0.0_f32; 256];
    let mut i = 0;
    while i < 256 {
        let c = i as f32 / 255.0;
        lut[i] = if c <= 0.04045 {
            c / 12.92
        } else {
            // Manual pow approximation for const context
            // (c + 0.055) / 1.055 raised to 2.4
            let base = (c + 0.055) / 1.055;
            // Use exp(2.4 * ln(base)) approximation via iteration
            let ln_base = const_ln(base);
            const_exp(2.4 * ln_base)
        };
        i += 1;
    }
    lut
};

/// Const-compatible natural log approximation
const fn const_ln(x: f32) -> f32 {
    // ln(x) using series expansion around 1
    // For x in [0.5, 1], use ln(x) = -ln(1/x)
    // We work with x in range ~0.05 to 1.0
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }

    // Reduce to range [0.5, 1] by extracting powers of 2
    let mut mantissa = x;
    let mut exp = 0_i32;
    while mantissa < 0.5 {
        mantissa *= 2.0;
        exp -= 1;
    }
    while mantissa > 1.0 {
        mantissa *= 0.5;
        exp += 1;
    }

    // ln(mantissa) where mantissa in [0.5, 1]
    // Use ln(1+u) series where u = mantissa - 1
    let u = mantissa - 1.0;
    let u2 = u * u;
    let u3 = u2 * u;
    let u4 = u3 * u;
    let u5 = u4 * u;
    let ln_m = u - u2/2.0 + u3/3.0 - u4/4.0 + u5/5.0;

    // ln(x) = ln(mantissa * 2^exp) = ln(mantissa) + exp * ln(2)
    ln_m + (exp as f32) * 0.693147180559945
}

/// Const-compatible exp approximation
const fn const_exp(x: f32) -> f32 {
    // exp(x) using Taylor series
    // Reduce range: exp(x) = exp(x - n*ln2) * 2^n
    let ln2 = 0.693147180559945_f32;
    let n = (x / ln2) as i32;
    let r = x - (n as f32) * ln2;

    // Taylor series for exp(r) where |r| < ln(2)
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r3 * r;
    let r5 = r4 * r;
    let exp_r = 1.0 + r + r2/2.0 + r3/6.0 + r4/24.0 + r5/120.0;

    // Multiply by 2^n
    let mut result = exp_r;
    let mut i = 0;
    if n > 0 {
        while i < n {
            result *= 2.0;
            i += 1;
        }
    } else {
        while i > n {
            result *= 0.5;
            i -= 1;
        }
    }
    result
}

/// Convert sRGB color component (0-255) to linear color space (0.0-1.0)
/// Uses precomputed lookup table for speed
#[inline]
fn srgb_to_linear(c: u8) -> f32 {
    SRGB_TO_LINEAR_LUT[c as usize]
}

/// Precomputed linear to sRGB lookup table (1024 entries for [0,1] range)
/// 10-bit precision is sufficient for 8-bit output
const LINEAR_TO_SRGB_LUT: [u8; 1024] = {
    let mut lut = [0_u8; 1024];
    let mut i = 0;
    while i < 1024 {
        let c = i as f32 / 1023.0;
        let v = if c <= 0.0031308 {
            12.92 * c
        } else {
            let ln_c = const_ln(c);
            1.055 * const_exp(ln_c / 2.4) - 0.055
        };
        lut[i] = if v <= 0.0 { 0 } else if v >= 1.0 { 255 } else { (v * 255.0 + 0.5) as u8 };
        i += 1;
    }
    lut
};

/// Convert linear color component (0.0-1.0) to sRGB (0-255)
/// Uses precomputed lookup table for speed
#[inline]
fn linear_to_srgb(c: f32) -> u8 {
    let idx = (c.clamp(0.0, 1.0) * 1023.0) as usize;
    LINEAR_TO_SRGB_LUT[idx]
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

    /// Get raw depth buffer for parallel access
    pub fn depth_buffer(&self) -> &[f32] {
        &self.depth
    }

    /// Merge another buffer into this one using depth testing.
    /// Used for combining parallel-rendered tile buffers.
    pub fn merge_from(&mut self, other: &PixelBuffer) {
        debug_assert_eq!(self.width, other.width);
        debug_assert_eq!(self.height, other.height);

        for i in 0..self.colors.len() {
            if other.colors[i].3 > 0 && other.depth[i] > self.depth[i] {
                self.colors[i] = other.colors[i];
                self.depth[i] = other.depth[i];
            }
        }
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

    /// Draw a shaded sphere - optimized version with fast pow approximations
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

        // Pre-convert base color to linear (0-1)
        let br = base_color.0 as f32 / 255.0;
        let bg = base_color.1 as f32 / 255.0;
        let bb = base_color.2 as f32 / 255.0;

        // Pre-computed light direction (normalized): upper-left-front
        const LIGHT: (f32, f32, f32) = (0.408, -0.511, 0.776);
        const HALF: (f32, f32, f32) = (0.230, -0.288, 1.0); // Unnormalized, will use dot directly

        let r2 = radius * radius;
        let inv_r = 1.0 / radius;
        let min_x = (cx - radius).floor() as i32;
        let max_x = (cx + radius).ceil() as i32;
        let min_y = (cy - radius).floor() as i32;
        let max_y = (cy + radius).ceil() as i32;

        for y in min_y..=max_y {
            let dy = (y as f32 + 0.5) - cy;
            let dy2 = dy * dy;

            for x in min_x..=max_x {
                let dx = (x as f32 + 0.5) - cx;
                let d2 = dx * dx + dy2;

                if d2 > r2 {
                    continue;
                }

                let dz = (r2 - d2).sqrt();
                let z = cz + dz;

                // Normal (already unit length since on sphere)
                let nx = dx * inv_r;
                let ny = dy * inv_r;
                let nz = dz * inv_r;

                // Diffuse (Lambert)
                let n_dot_l = (nx * LIGHT.0 + ny * LIGHT.1 + nz * LIGHT.2).max(0.0);

                // Fast specular: (n·h)^32 ≈ x^4^4^2 using repeated squaring
                let n_dot_h = (nx * HALF.0 + ny * HALF.1 + nz * HALF.2).max(0.0);
                let spec2 = n_dot_h * n_dot_h;
                let spec4 = spec2 * spec2;
                let spec8 = spec4 * spec4;
                let spec16 = spec8 * spec8;
                let spec32 = spec16 * spec16;
                let specular = if n_dot_l > 0.0 { 0.35 * spec32 } else { 0.0 };

                // Simple rim: (1-nz)^2
                let rim_t = 1.0 - nz;
                let rim = 0.15 * rim_t * rim_t;

                // Combine: ambient + diffuse + specular + rim
                let shade = 0.15 + 0.70 * n_dot_l;
                let r = (br * shade + specular + rim * br).min(1.0);
                let g = (bg * shade + specular + rim * bg).min(1.0);
                let b = (bb * shade + specular + rim * bb).min(1.0);

                let color = (
                    (r * 255.0) as u8,
                    (g * 255.0) as u8,
                    (b * 255.0) as u8,
                );

                self.set_pixel(x, y, z, color);
            }
        }
    }

    /// Draw a shaded cylinder - fast version using bounding box raycast
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
        let length_sq = dx * dx + dy * dy + dz * dz;

        if length_sq < 0.01 {
            return;
        }

        let length = length_sq.sqrt();
        let inv_len = 1.0 / length;

        // Cylinder axis direction
        let ax = dx * inv_len;
        let ay = dy * inv_len;
        let az = dz * inv_len;

        // Pre-convert color
        let br = color.0 as f32 / 255.0;
        let bg = color.1 as f32 / 255.0;
        let bb = color.2 as f32 / 255.0;

        // Light direction (pre-normalized)
        const LIGHT: (f32, f32, f32) = (0.408, -0.511, 0.776);

        // 2D bounding box in screen space
        let min_x = (x0.min(x1) - radius).floor() as i32;
        let max_x = (x0.max(x1) + radius).ceil() as i32;
        let min_y = (y0.min(y1) - radius).floor() as i32;
        let max_y = (y0.max(y1) + radius).ceil() as i32;

        let r2 = radius * radius;

        for y in min_y..=max_y {
            let py = y as f32 + 0.5;

            for x in min_x..=max_x {
                let px = x as f32 + 0.5;

                // Vector from p0 to pixel
                let vx = px - x0;
                let vy = py - y0;

                // Project onto cylinder axis: t = (v · axis)
                let t = (vx * ax + vy * ay).clamp(0.0, length);

                // Closest point on axis to pixel
                let closest_x = x0 + ax * t;
                let closest_y = y0 + ay * t;
                let closest_z = z0 + az * t;

                // Distance from pixel to axis (in 2D)
                let dist_x = px - closest_x;
                let dist_y = py - closest_y;
                let dist_sq = dist_x * dist_x + dist_y * dist_y;

                if dist_sq > r2 {
                    continue;
                }

                // Calculate z on cylinder surface
                let dist = dist_sq.sqrt();
                let surface_z = if dist < radius {
                    closest_z + (r2 - dist_sq).sqrt() * 0.3
                } else {
                    closest_z
                };

                // Normal: points from axis to surface point
                let (nx, ny, nz) = if dist > 0.001 {
                    let inv_d = 1.0 / dist;
                    let nz_contrib = if dist < radius { (r2 - dist_sq).sqrt() / radius } else { 0.0 };
                    (dist_x * inv_d * (1.0 - nz_contrib), dist_y * inv_d * (1.0 - nz_contrib), nz_contrib)
                } else {
                    (0.0, 0.0, 1.0)
                };

                // Diffuse lighting
                let n_dot_l = (nx * LIGHT.0 + ny * LIGHT.1 + nz * LIGHT.2).max(0.0);
                let shade = 0.20 + 0.65 * n_dot_l;

                // Simple specular
                let spec = if n_dot_l > 0.0 && nz > 0.3 {
                    let spec_t = nz * nz;
                    0.2 * spec_t * spec_t
                } else {
                    0.0
                };

                let r = ((br * shade + spec).min(1.0) * 255.0) as u8;
                let g = ((bg * shade + spec).min(1.0) * 255.0) as u8;
                let b = ((bb * shade + spec).min(1.0) * 255.0) as u8;

                self.set_pixel(x, y, surface_z, (r, g, b));
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

/// Fast SSAO - just 4 cardinal samples for performance
/// Apply Screen Space Ambient Occlusion for enhanced depth perception.
/// Optimized version using only 4 samples for terminal rendering.
pub fn apply_ssao(buffer: &mut PixelBuffer, radius: f32, strength: f32) {
    let width = buffer.width;
    let height = buffer.height;

    if width < 5 || height < 5 {
        return;
    }

    // Sample radius scaled to image size
    let r = (radius * (width.min(height) as f32 / 200.0).max(1.0)).max(2.0) as i32;

    // Compute occlusion factors in parallel - simple 4-direction sampling
    let occlusion: Vec<f32> = (0..height)
        .into_par_iter()
        .flat_map(|y| {
            let mut row = vec![0.0_f32; width];
            let yi = y as i32;

            for x in 0..width {
                let xi = x as i32;
                let idx = y * width + x;

                // Skip background pixels
                if buffer.colors[idx].3 == 0 {
                    continue;
                }

                let center_depth = buffer.depth[idx];
                if center_depth <= f32::NEG_INFINITY {
                    continue;
                }

                // Sample 4 cardinal directions
                let mut occluded = 0_u8;
                let offsets: [(i32, i32); 4] = [(r, 0), (-r, 0), (0, r), (0, -r)];

                for (dx, dy) in offsets {
                    let sx = xi + dx;
                    let sy = yi + dy;

                    if sx >= 0 && sy >= 0 && (sx as usize) < width && (sy as usize) < height {
                        let s_idx = sy as usize * width + sx as usize;
                        if buffer.colors[s_idx].3 > 0 {
                            let sample_depth = buffer.depth[s_idx];
                            // If sample is in front, we're occluded
                            if sample_depth > center_depth + 0.5 {
                                occluded += 1;
                            }
                        }
                    }
                }

                if occluded > 0 {
                    row[x] = (occluded as f32 * 0.15 * strength).min(0.4);
                }
            }

            row
        })
        .collect();

    // Apply darkening - simple multiply, skip sRGB conversion for speed
    buffer.colors
        .par_iter_mut()
        .zip(occlusion.par_iter())
        .for_each(|(color, &occ)| {
            if occ > 0.0 {
                let darken = 1.0 - occ;
                *color = (
                    (color.0 as f32 * darken) as u8,
                    (color.1 as f32 * darken) as u8,
                    (color.2 as f32 * darken) as u8,
                    color.3,
                );
            }
        });
}

/// Fast filmic tone curve applied directly to sRGB values
/// Approximates ACES look without expensive color space conversions
#[inline]
fn fast_tonemap(x: u8, exposure: f32) -> u8 {
    // Work in 0-1 range
    let v = x as f32 / 255.0 * exposure;
    // Simple S-curve: slight contrast boost with soft highlight rolloff
    // Cheaper than full ACES but gives similar feel
    let t = if v < 0.5 {
        v * v * 2.0  // Darken shadows slightly
    } else {
        1.0 - (1.0 - v) * (1.0 - v) * 2.0  // Soft highlights
    };
    (t.clamp(0.0, 1.0) * 255.0) as u8
}

/// Apply fast tone mapping to the buffer for improved color reproduction.
/// Optimized version that works directly on sRGB values.
pub fn apply_tone_mapping(buffer: &mut PixelBuffer, exposure: f32) {
    buffer.colors
        .par_iter_mut()
        .for_each(|color| {
            if color.3 == 0 {
                return;
            }

            *color = (
                fast_tonemap(color.0, exposure),
                fast_tonemap(color.1, exposure),
                fast_tonemap(color.2, exposure),
                color.3,
            );
        });
}
