//! Pixel buffer for raster rendering
//!
//! Stores per-pixel color and depth for image and half-block outputs.

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const SUBPIXEL_Q: i32 = 4; // Quarter-pixel offsets for LUT reuse.
const RADIUS_Q: i32 = 4; // Quarter-pixel radius quantization.
const CYLINDER_PROFILE_SAMPLES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SphereLutKey {
    radius_q: i32,
    sub_x: u8,
    sub_y: u8,
}

#[derive(Debug, Clone)]
struct SphereLutRow {
    dy: i32,
    x_start: i32,
    len: usize,
    offset: usize,
}

#[derive(Debug, Clone)]
struct SphereLut {
    rows: Vec<SphereLutRow>,
    z: Vec<f32>,
    base_mul: Vec<f32>,
    spec: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CylinderProfileKey {
    radius_q: i32,
}

#[derive(Debug, Clone)]
struct CylinderProfile {
    radius: f32,
    inv_r2: f32,
    nz: Vec<f32>,
    radial: Vec<f32>,
    inv_dist: Vec<f32>,
    spec: Vec<f32>,
    z_offset: Vec<f32>,
}

fn sphere_lut_cache() -> &'static Mutex<HashMap<SphereLutKey, Arc<SphereLut>>> {
    static CACHE: OnceLock<Mutex<HashMap<SphereLutKey, Arc<SphereLut>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cylinder_profile_cache() -> &'static Mutex<HashMap<CylinderProfileKey, Arc<CylinderProfile>>> {
    static CACHE: OnceLock<Mutex<HashMap<CylinderProfileKey, Arc<CylinderProfile>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_sphere_lut(key: SphereLutKey) -> Arc<SphereLut> {
    {
        let cache = sphere_lut_cache();
        let map = cache.lock().unwrap();
        if let Some(lut) = map.get(&key) {
            return Arc::clone(lut);
        }
    }

    let lut = Arc::new(build_sphere_lut(key));
    let cache = sphere_lut_cache();
    let mut map = cache.lock().unwrap();
    map.insert(key, Arc::clone(&lut));
    lut
}

fn get_cylinder_profile(key: CylinderProfileKey) -> Arc<CylinderProfile> {
    {
        let cache = cylinder_profile_cache();
        let map = cache.lock().unwrap();
        if let Some(profile) = map.get(&key) {
            return Arc::clone(profile);
        }
    }

    let profile = Arc::new(build_cylinder_profile(key));
    let cache = cylinder_profile_cache();
    let mut map = cache.lock().unwrap();
    map.insert(key, Arc::clone(&profile));
    profile
}

fn quantize_subpixel(value: f32) -> (i32, u8) {
    let mut base = value.floor() as i32;
    let frac = value - base as f32;
    let mut q = ((frac * SUBPIXEL_Q as f32) + 0.5).floor() as i32;
    if q >= SUBPIXEL_Q {
        q = 0;
        base += 1;
    }
    (base, q as u8)
}

fn quantize_radius(radius: f32) -> i32 {
    (radius * RADIUS_Q as f32).round().max(1.0) as i32
}

fn build_sphere_lut(key: SphereLutKey) -> SphereLut {
    // 3-light rig for professional look (like PyMOL/ChimeraX):
    // Key light: main directional light from upper-left-front
    // Fill light: softer light from lower-right to fill shadows
    // Rim light: backlight for edge definition

    // Key light (warm, strong) - upper left front (normalized)
    const KEY_DIR: (f32, f32, f32) = (0.408, -0.511, 0.776);
    // Half-vector for Blinn-Phong: H = normalize(L + V), V = (0,0,1)
    const KEY_HALF: (f32, f32, f32) = (0.216, -0.270, 0.938);
    const KEY_INTENSITY: f32 = 0.65;

    // Fill light (cool, softer) - lower right
    const FILL_DIR: (f32, f32, f32) = (-0.5, 0.3, 0.81);
    const FILL_INTENSITY: f32 = 0.25;

    // Rim/back light - from behind for edge highlights
    const RIM_DIR: (f32, f32, f32) = (0.0, 0.2, -0.98);
    const RIM_INTENSITY: f32 = 0.3;

    let radius = key.radius_q as f32 / RADIUS_Q as f32;
    let r2 = radius * radius;
    let r_int = radius.ceil().max(1.0) as i32;
    let off_x = key.sub_x as f32 / SUBPIXEL_Q as f32;
    let off_y = key.sub_y as f32 / SUBPIXEL_Q as f32;

    let mut rows = Vec::new();
    let mut z = Vec::new();
    let mut base_mul = Vec::new();
    let mut spec = Vec::new();

    let inv_r = 1.0 / radius;
    for dy_i in -r_int..=r_int {
        let dy = dy_i as f32 - off_y;
        let dy2 = dy * dy;
        if dy2 > r2 {
            continue;
        }

        let max_dx = (r2 - dy2).sqrt();
        let x_start = (off_x - max_dx).ceil() as i32;
        let x_end = (off_x + max_dx).floor() as i32;
        if x_end < x_start {
            continue;
        }

        let len = (x_end - x_start + 1) as usize;
        let offset = z.len();
        rows.push(SphereLutRow {
            dy: dy_i,
            x_start,
            len,
            offset,
        });

        for dx_i in x_start..=x_end {
            let dx = dx_i as f32 - off_x;
            let d2 = dx * dx + dy2;
            let dz = (r2 - d2).max(0.0).sqrt();
            let nx = dx * inv_r;
            let ny = dy * inv_r;
            let nz = dz * inv_r;

            // Key light contribution (with specular)
            let key_dot = (nx * KEY_DIR.0 + ny * KEY_DIR.1 + nz * KEY_DIR.2).max(0.0);
            let key_diffuse = KEY_INTENSITY * key_dot;

            // Key specular (Blinn-Phong)
            let n_dot_h = (nx * KEY_HALF.0 + ny * KEY_HALF.1 + nz * KEY_HALF.2).max(0.0);
            let spec2 = n_dot_h * n_dot_h;
            let spec4 = spec2 * spec2;
            let spec8 = spec4 * spec4;
            let spec16 = spec8 * spec8;
            let spec32 = spec16 * spec16;
            let key_spec = if key_dot > 0.0 { 0.4 * spec32 } else { 0.0 };

            // Fill light contribution (no specular, just soft fill)
            let fill_dot = (nx * FILL_DIR.0 + ny * FILL_DIR.1 + nz * FILL_DIR.2).max(0.0);
            let fill_diffuse = FILL_INTENSITY * fill_dot;

            // Rim light - Fresnel-like effect for edge highlighting
            let rim_dot = (nx * RIM_DIR.0 + ny * RIM_DIR.1 + nz * RIM_DIR.2).max(0.0);
            let fresnel = 1.0 - nz; // Stronger at edges
            let rim_contrib = RIM_INTENSITY * rim_dot * fresnel * fresnel;

            // Ambient occlusion approximation - darken edges where nz is low
            let edge_darken = (1.0 - nz) * 0.12;

            // Combine all lighting
            let ambient = 0.15;
            let total_diffuse = ambient + key_diffuse + fill_diffuse + rim_contrib - edge_darken;
            let base = total_diffuse.min(1.0);

            // Combined specular
            let specular = key_spec;

            z.push(dz);
            base_mul.push(base);
            spec.push(specular);
        }
    }

    SphereLut {
        rows,
        z,
        base_mul,
        spec,
    }
}

fn build_cylinder_profile(key: CylinderProfileKey) -> CylinderProfile {
    let radius = key.radius_q as f32 / RADIUS_Q as f32;
    let inv_r2 = 1.0 / (radius * radius);
    let mut nz = Vec::with_capacity(CYLINDER_PROFILE_SAMPLES);
    let mut radial = Vec::with_capacity(CYLINDER_PROFILE_SAMPLES);
    let mut inv_dist = Vec::with_capacity(CYLINDER_PROFILE_SAMPLES);
    let mut spec = Vec::with_capacity(CYLINDER_PROFILE_SAMPLES);
    let mut z_offset = Vec::with_capacity(CYLINDER_PROFILE_SAMPLES);

    for i in 0..CYLINDER_PROFILE_SAMPLES {
        let t = if CYLINDER_PROFILE_SAMPLES > 1 {
            i as f32 / (CYLINDER_PROFILE_SAMPLES - 1) as f32
        } else {
            0.0
        };
        let nz_val = (1.0 - t).max(0.0).sqrt();
        // radial component: sqrt(1 - nz²) = sqrt(t) for unit normal
        let radial_val = t.sqrt();
        let inv_dist_val = if t > 0.0 {
            1.0 / (radius * t.sqrt())
        } else {
            0.0
        };
        let spec_val = 0.2 * (nz_val * nz_val).powi(2);
        let z_val = nz_val * radius * 0.3;

        nz.push(nz_val);
        radial.push(radial_val);
        inv_dist.push(inv_dist_val);
        spec.push(spec_val);
        z_offset.push(z_val);
    }

    CylinderProfile {
        radius,
        inv_r2,
        nz,
        radial,
        inv_dist,
        spec,
        z_offset,
    }
}

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
    ln_m + (exp as f32) * std::f32::consts::LN_2
}

/// Const-compatible exp approximation
const fn const_exp(x: f32) -> f32 {
    // exp(x) using Taylor series
    // Reduce range: exp(x) = exp(x - n*ln2) * 2^n
    let ln2 = std::f32::consts::LN_2;
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

    /// Fill the buffer with a solid background color
    pub fn fill_background(&mut self, color: (u8, u8, u8)) {
        self.colors.fill((color.0, color.1, color.2, 255));
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

    /// Copy a smaller buffer into this one at a given offset.
    pub fn blit_from(&mut self, src: &PixelBuffer, dst_x: usize, dst_y: usize) {
        debug_assert!(dst_x + src.width <= self.width);
        debug_assert!(dst_y + src.height <= self.height);

        for y in 0..src.height {
            let dst_row = (dst_y + y) * self.width + dst_x;
            let src_row = y * src.width;
            for x in 0..src.width {
                let src_idx = src_row + x;
                let dst_idx = dst_row + x;
                let src_color = src.colors[src_idx];
                if src_color.3 == 0 {
                    continue;
                }
                let src_depth = src.depth[src_idx];
                if src_depth > self.depth[dst_idx] {
                    self.colors[dst_idx] = src_color;
                    self.depth[dst_idx] = src_depth;
                }
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

    /// Draw a filled circle (flat shading) - LUT-based for performance.
    /// Reuses sphere LUT span structure but with flat color (no shading math).
    pub fn draw_circle(
        &mut self,
        cx: f32,
        cy: f32,
        cz: f32,
        radius: f32,
        color: (u8, u8, u8),
    ) {
        if radius < 1.2 {
            self.set_pixel(cx as i32, cy as i32, cz, color);
            return;
        }

        // Reuse sphere LUT infrastructure for span-based circle rendering
        // This avoids per-pixel sqrt and leverages cached LUT data
        let radius_q = quantize_radius(radius);
        let (cx_base, sub_x) = quantize_subpixel(cx);
        let (cy_base, sub_y) = quantize_subpixel(cy);
        let lut = get_sphere_lut(SphereLutKey { radius_q, sub_x, sub_y });

        let width = self.width as i32;
        let height = self.height as i32;
        let color_rgba = (color.0, color.1, color.2, 255u8);

        for row in &lut.rows {
            let y = cy_base + row.dy;
            if y < 0 || y >= height {
                continue;
            }

            let mut x0 = cx_base + row.x_start;
            let mut x1 = x0 + row.len as i32;
            if x1 <= 0 || x0 >= width {
                continue;
            }

            let mut skip = 0usize;
            if x0 < 0 {
                skip = (-x0) as usize;
                x0 = 0;
            }
            if x1 > width {
                x1 = width;
            }
            if x0 >= x1 {
                continue;
            }

            let mut src_idx = row.offset + skip;
            let mut dst_idx = y as usize * self.width + x0 as usize;
            let count = (x1 - x0) as usize;

            // Use LUT z values scaled down for flat circle depth (similar curve to old impl)
            for _ in 0..count {
                let z = cz + lut.z[src_idx] * 0.5;
                if z > self.depth[dst_idx] {
                    self.colors[dst_idx] = color_rgba;
                    self.depth[dst_idx] = z;
                }
                src_idx += 1;
                dst_idx += 1;
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

        let radius_q = quantize_radius(radius);
        let (cx_base, sub_x) = quantize_subpixel(cx);
        let (cy_base, sub_y) = quantize_subpixel(cy);
        let lut = get_sphere_lut(SphereLutKey {
            radius_q,
            sub_x,
            sub_y,
        });

        let br = base_color.0 as f32 / 255.0;
        let bg = base_color.1 as f32 / 255.0;
        let bb = base_color.2 as f32 / 255.0;
        let width = self.width as i32;
        let height = self.height as i32;

        for row in &lut.rows {
            let y = cy_base + row.dy;
            if y < 0 || y >= height {
                continue;
            }

            let mut x0 = cx_base + row.x_start;
            let mut x1 = x0 + row.len as i32;
            if x1 <= 0 || x0 >= width {
                continue;
            }

            let mut skip = 0usize;
            if x0 < 0 {
                skip = (-x0) as usize;
                x0 = 0;
            }
            if x1 > width {
                x1 = width;
            }
            if x0 >= x1 {
                continue;
            }

            let mut src_idx = row.offset + skip;
            let mut dst_idx = y as usize * self.width + x0 as usize;
            let count = (x1 - x0) as usize;

            for _ in 0..count {
                let z = cz + lut.z[src_idx];
                if z > self.depth[dst_idx] {
                    let base = lut.base_mul[src_idx];
                    let spec = lut.spec[src_idx];
                    let r = (br * base + spec).min(1.0);
                    let g = (bg * base + spec).min(1.0);
                    let b = (bb * base + spec).min(1.0);
                    self.colors[dst_idx] = ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255);
                    self.depth[dst_idx] = z;
                }
                src_idx += 1;
                dst_idx += 1;
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
        let len2 = dx * dx + dy * dy;
        if len2 < 0.001 {
            let cx = (x0 + x1) * 0.5;
            let cy = (y0 + y1) * 0.5;
            let cz = z0.max(z1);
            self.draw_sphere_shaded(cx, cy, cz, radius, color);
            return;
        }
        let inv_len2 = 1.0 / len2;

        let radius_q = quantize_radius(radius);
        let profile = get_cylinder_profile(CylinderProfileKey { radius_q });
        let r = profile.radius;
        let r2 = r * r;

        let br = color.0 as f32 / 255.0;
        let bg = color.1 as f32 / 255.0;
        let bb = color.2 as f32 / 255.0;

        // 3-light rig matching sphere shading (normalized vectors)
        const KEY_DIR: (f32, f32, f32) = (0.408, -0.511, 0.776);
        const KEY_HALF: (f32, f32, f32) = (0.216, -0.270, 0.938);
        const FILL_DIR: (f32, f32, f32) = (-0.5, 0.3, 0.81);
        const RIM_DIR: (f32, f32, f32) = (0.0, 0.2, -0.98);

        let width = self.width as i32;
        let height = self.height as i32;
        let mut min_x = (x0.min(x1) - r).floor() as i32;
        let mut max_x = (x0.max(x1) + r).ceil() as i32;
        let mut min_y = (y0.min(y1) - r).floor() as i32;
        let mut max_y = (y0.max(y1) + r).ceil() as i32;

        min_x = min_x.max(0);
        min_y = min_y.max(0);
        max_x = max_x.min(width - 1);
        max_y = max_y.min(height - 1);
        if min_x > max_x || min_y > max_y {
            return;
        }

        for y in min_y..=max_y {
            let py = y as f32 + 0.5;
            for x in min_x..=max_x {
                let px = x as f32 + 0.5;
                let vx = px - x0;
                let vy = py - y0;
                let t_axis = (vx * dx + vy * dy) * inv_len2;
                let t_axis = t_axis.clamp(0.0, 1.0);

                let closest_x = x0 + dx * t_axis;
                let closest_y = y0 + dy * t_axis;

                let dist_x = px - closest_x;
                let dist_y = py - closest_y;
                let dist_sq = dist_x * dist_x + dist_y * dist_y;
                if dist_sq > r2 {
                    continue;
                }

                let t_rad = (dist_sq * profile.inv_r2).min(1.0);
                let idx = ((t_rad * (CYLINDER_PROFILE_SAMPLES as f32 - 1.0)) as usize)
                    .min(CYLINDER_PROFILE_SAMPLES - 1);

                let nz = profile.nz[idx];
                let radial = profile.radial[idx];
                let inv_dist = profile.inv_dist[idx];
                let nx = dist_x * inv_dist * radial;
                let ny = dist_y * inv_dist * radial;

                // Key light (main diffuse + specular)
                let key_dot = (nx * KEY_DIR.0 + ny * KEY_DIR.1 + nz * KEY_DIR.2).max(0.0);
                let key_diffuse = 0.65 * key_dot;

                // Key specular
                let n_dot_h = (nx * KEY_HALF.0 + ny * KEY_HALF.1 + nz * KEY_HALF.2).max(0.0);
                let spec_pow = n_dot_h * n_dot_h;
                let spec_pow = spec_pow * spec_pow; // ^4
                let spec_pow = spec_pow * spec_pow; // ^8
                let spec_pow = spec_pow * spec_pow; // ^16
                let spec_pow = spec_pow * spec_pow; // ^32
                let spec = if key_dot > 0.0 && nz > 0.2 { 0.35 * spec_pow } else { 0.0 };

                // Fill light (soft fill from opposite side)
                let fill_dot = (nx * FILL_DIR.0 + ny * FILL_DIR.1 + nz * FILL_DIR.2).max(0.0);
                let fill_diffuse = 0.25 * fill_dot;

                // Rim light (edge highlighting)
                let rim_dot = (nx * RIM_DIR.0 + ny * RIM_DIR.1 + nz * RIM_DIR.2).max(0.0);
                let fresnel = 1.0 - nz;
                let rim = 0.25 * rim_dot * fresnel * fresnel;

                // Combine lighting
                let ambient = 0.12;
                let shade = ambient + key_diffuse + fill_diffuse + rim + nz * 0.1;

                let surface_z = z0 + (z1 - z0) * t_axis + profile.z_offset[idx];

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
                let blend = (1.0 - fx) * 0.5;
                let aa_color = blend_color(color, blend);
                self.set_pixel(xi + 1, yi, z, aa_color);
            }
            if fy > 0.3 {
                let blend = (1.0 - fy) * 0.5;
                let aa_color = blend_color(color, blend);
                self.set_pixel(xi, yi + 1, z, aa_color);
            }
        }
    }

    /// Draw a flat ribbon/plank for beta sheets - uses filled rectangle approach
    /// to avoid fanning artifacts on curves
    pub fn draw_flat_sheet(
        &mut self,
        x0: f32,
        y0: f32,
        z0: f32,
        x1: f32,
        y1: f32,
        z1: f32,
        width: f32,
        color: (u8, u8, u8),
    ) {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let length = (dx * dx + dy * dy).sqrt();
        if length < 0.5 {
            return;
        }

        // Use axis-aligned bounding box rasterization
        let half_w = width * 0.5;

        // Compute the 4 corners of the ribbon quad
        let perp_x = -dy / length;
        let perp_y = dx / length;

        let c0x = x0 - perp_x * half_w;
        let c0y = y0 - perp_y * half_w;
        let c1x = x0 + perp_x * half_w;
        let c1y = y0 + perp_y * half_w;
        let c2x = x1 + perp_x * half_w;
        let c2y = y1 + perp_y * half_w;
        let c3x = x1 - perp_x * half_w;
        let c3y = y1 - perp_y * half_w;

        // Bounding box
        let min_x = c0x.min(c1x).min(c2x).min(c3x).floor() as i32;
        let max_x = c0x.max(c1x).max(c2x).max(c3x).ceil() as i32;
        let min_y = c0y.min(c1y).min(c2y).min(c3y).floor() as i32;
        let max_y = c0y.max(c1y).max(c2y).max(c3y).ceil() as i32;

        // Shading setup
        const KEY_DIR: (f32, f32, f32) = (0.408, -0.511, 0.776);
        const FILL_DIR: (f32, f32, f32) = (-0.5, 0.3, 0.81);
        let nz = 1.0_f32;
        let key_dot = (nz * KEY_DIR.2).max(0.0);
        let fill_dot = (nz * FILL_DIR.2).max(0.0);
        let ambient = 0.18;
        let center_shade = ambient + 0.65 * key_dot + 0.25 * fill_dot;
        let edge_shade = center_shade * 0.75;

        let br = color.0 as f32 / 255.0;
        let bg = color.1 as f32 / 255.0;
        let bb = color.2 as f32 / 255.0;

        // Direction vector (normalized)
        let dir_x = dx / length;
        let dir_y = dy / length;

        // Rasterize using point-in-quad test
        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let fx = px as f32 + 0.5;
                let fy = py as f32 + 0.5;

                // Project point onto ribbon axis
                let to_pt_x = fx - x0;
                let to_pt_y = fy - y0;

                // Distance along the ribbon
                let along = to_pt_x * dir_x + to_pt_y * dir_y;
                if along < -0.5 || along > length + 0.5 {
                    continue;
                }

                // Distance perpendicular to ribbon
                let perp_dist = (to_pt_x * perp_x + to_pt_y * perp_y).abs();
                if perp_dist > half_w + 0.5 {
                    continue;
                }

                // Calculate z by interpolating along the ribbon
                let t = (along / length).clamp(0.0, 1.0);
                let z = z0 + (z1 - z0) * t;

                // Edge shading based on perpendicular distance
                let edge_factor = 1.0 - (perp_dist / half_w).min(1.0);
                let shade = edge_shade + (center_shade - edge_shade) * edge_factor;

                let r = (br * shade).min(1.0);
                let g = (bg * shade).min(1.0);
                let b = (bb * shade).min(1.0);

                self.set_pixel(
                    px,
                    py,
                    z,
                    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8),
                );
            }
        }
    }

    /// Draw a 3D arrow for beta sheet termini - uses triangle rasterization
    pub fn draw_sheet_arrow(
        &mut self,
        tip_x: f32,
        tip_y: f32,
        tip_z: f32,
        dir_x: f32,
        dir_y: f32,
        arrow_length: f32,
        arrow_width: f32,
        color: (u8, u8, u8),
    ) {
        let dir_len = (dir_x * dir_x + dir_y * dir_y).sqrt();
        if dir_len < 0.001 {
            return;
        }

        let dx = dir_x / dir_len;
        let dy = dir_y / dir_len;
        let perp_x = -dy;
        let perp_y = dx;

        // Arrow base position
        let base_x = tip_x - dx * arrow_length;
        let base_y = tip_y - dy * arrow_length;
        let half_w = arrow_width * 0.5;

        // Triangle vertices: tip and two base corners
        let v0 = (tip_x, tip_y); // tip
        let v1 = (base_x - perp_x * half_w, base_y - perp_y * half_w); // left base
        let v2 = (base_x + perp_x * half_w, base_y + perp_y * half_w); // right base

        // Bounding box
        let min_x = v0.0.min(v1.0).min(v2.0).floor() as i32;
        let max_x = v0.0.max(v1.0).max(v2.0).ceil() as i32;
        let min_y = v0.1.min(v1.1).min(v2.1).floor() as i32;
        let max_y = v0.1.max(v1.1).max(v2.1).ceil() as i32;

        // Shading setup
        const KEY_DIR: (f32, f32, f32) = (0.408, -0.511, 0.776);
        const FILL_DIR: (f32, f32, f32) = (-0.5, 0.3, 0.81);
        let nz = 1.0_f32;
        let key_dot = (nz * KEY_DIR.2).max(0.0);
        let fill_dot = (nz * FILL_DIR.2).max(0.0);
        let ambient = 0.18;
        let center_shade = ambient + 0.65 * key_dot + 0.25 * fill_dot;
        let edge_shade = center_shade * 0.75;

        let br = color.0 as f32 / 255.0;
        let bg = color.1 as f32 / 255.0;
        let bb = color.2 as f32 / 255.0;

        // Helper: sign of cross product for point-in-triangle test
        fn sign(p1: (f32, f32), p2: (f32, f32), p3: (f32, f32)) -> f32 {
            (p1.0 - p3.0) * (p2.1 - p3.1) - (p2.0 - p3.0) * (p1.1 - p3.1)
        }

        // Rasterize triangle
        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let pt = (px as f32 + 0.5, py as f32 + 0.5);

                // Point-in-triangle test using barycentric coordinates
                let d1 = sign(pt, v0, v1);
                let d2 = sign(pt, v1, v2);
                let d3 = sign(pt, v2, v0);

                let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
                let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);

                if has_neg && has_pos {
                    continue; // Outside triangle
                }

                // Calculate distance from center axis for shading
                let to_pt_x = pt.0 - base_x;
                let to_pt_y = pt.1 - base_y;
                let perp_dist = (to_pt_x * perp_x + to_pt_y * perp_y).abs();
                let along = to_pt_x * dx + to_pt_y * dy;
                let max_width_at_pos = half_w * (1.0 - (along / arrow_length).clamp(0.0, 1.0));
                let edge_factor = if max_width_at_pos > 0.01 {
                    1.0 - (perp_dist / max_width_at_pos).min(1.0)
                } else {
                    1.0
                };

                let shade = edge_shade + (center_shade - edge_shade) * edge_factor;
                let r = (br * shade).min(1.0);
                let g = (bg * shade).min(1.0);
                let b = (bb * shade).min(1.0);

                self.set_pixel(
                    px,
                    py,
                    tip_z,
                    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8),
                );
            }
        }
    }

    /// Draw a filled triangle with smooth shading using interpolated normals
    /// Uses barycentric coordinates for interpolation
    pub fn draw_triangle_shaded(
        &mut self,
        // Vertex positions (screen space)
        x0: f32, y0: f32, z0: f32,
        x1: f32, y1: f32, z1: f32,
        x2: f32, y2: f32, z2: f32,
        // Vertex normals (for lighting)
        nx0: f32, ny0: f32, nz0: f32,
        nx1: f32, ny1: f32, nz1: f32,
        nx2: f32, ny2: f32, nz2: f32,
        // Base color
        color: (u8, u8, u8),
    ) {
        // 3-light rig matching sphere/cylinder shading
        const KEY_DIR: (f32, f32, f32) = (0.408, -0.511, 0.776);
        const KEY_HALF: (f32, f32, f32) = (0.216, -0.270, 0.938);
        const FILL_DIR: (f32, f32, f32) = (-0.5, 0.3, 0.81);
        const RIM_DIR: (f32, f32, f32) = (0.0, 0.2, -0.98);

        // Bounding box
        let min_x = x0.min(x1).min(x2).floor() as i32;
        let max_x = x0.max(x1).max(x2).ceil() as i32;
        let min_y = y0.min(y1).min(y2).floor() as i32;
        let max_y = y0.max(y1).max(y2).ceil() as i32;

        let width = self.width as i32;
        let height = self.height as i32;

        // Clamp to screen bounds
        let min_x = min_x.max(0);
        let max_x = max_x.min(width - 1);
        let min_y = min_y.max(0);
        let max_y = max_y.min(height - 1);

        if min_x > max_x || min_y > max_y {
            return;
        }

        // Edge function denominator (2x signed area of triangle)
        let area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if area.abs() < 0.001 {
            return; // Degenerate triangle
        }
        let inv_area = 1.0 / area;

        let br = color.0 as f32 / 255.0;
        let bg = color.1 as f32 / 255.0;
        let bb = color.2 as f32 / 255.0;

        // Rasterize using edge functions
        for py in min_y..=max_y {
            let fy = py as f32 + 0.5;
            for px in min_x..=max_x {
                let fx = px as f32 + 0.5;

                // Compute barycentric coordinates
                let w0 = ((x1 - fx) * (y2 - fy) - (x2 - fx) * (y1 - fy)) * inv_area;
                let w1 = ((x2 - fx) * (y0 - fy) - (x0 - fx) * (y2 - fy)) * inv_area;
                let w2 = 1.0 - w0 - w1;

                // Check if point is inside triangle
                if w0 < -0.001 || w1 < -0.001 || w2 < -0.001 {
                    continue;
                }

                // Interpolate z
                let z = w0 * z0 + w1 * z1 + w2 * z2;

                // Depth test
                let idx = py as usize * self.width + px as usize;
                if z <= self.depth[idx] {
                    continue;
                }

                // Interpolate normal
                let nx = w0 * nx0 + w1 * nx1 + w2 * nx2;
                let ny = w0 * ny0 + w1 * ny1 + w2 * ny2;
                let nz = w0 * nz0 + w1 * nz1 + w2 * nz2;

                // Normalize
                let len = (nx * nx + ny * ny + nz * nz).sqrt();
                let (nx, ny, nz) = if len > 0.001 {
                    (nx / len, ny / len, nz / len)
                } else {
                    (0.0, 0.0, 1.0)
                };

                // Key light diffuse
                let key_dot = (nx * KEY_DIR.0 + ny * KEY_DIR.1 + nz * KEY_DIR.2).max(0.0);
                let key_diffuse = 0.65 * key_dot;

                // Key specular (Blinn-Phong)
                let n_dot_h = (nx * KEY_HALF.0 + ny * KEY_HALF.1 + nz * KEY_HALF.2).max(0.0);
                let spec_pow = n_dot_h.powi(32);
                let spec = if key_dot > 0.0 { 0.3 * spec_pow } else { 0.0 };

                // Fill light
                let fill_dot = (nx * FILL_DIR.0 + ny * FILL_DIR.1 + nz * FILL_DIR.2).max(0.0);
                let fill_diffuse = 0.25 * fill_dot;

                // Rim light
                let rim_dot = (nx * RIM_DIR.0 + ny * RIM_DIR.1 + nz * RIM_DIR.2).max(0.0);
                let fresnel = (1.0 - nz.abs()).max(0.0);
                let rim = 0.2 * rim_dot * fresnel * fresnel;

                // Ambient
                let ambient = 0.15;
                let shade = ambient + key_diffuse + fill_diffuse + rim;

                let r = ((br * shade + spec).min(1.0) * 255.0) as u8;
                let g = ((bg * shade + spec).min(1.0) * 255.0) as u8;
                let b = ((bb * shade + spec).min(1.0) * 255.0) as u8;

                self.colors[idx] = (r, g, b, 255);
                self.depth[idx] = z;
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
    let mut dst = PixelBuffer::new(dst_width, dst_height);

    dst.colors
        .par_chunks_mut(dst_width)
        .zip(dst.depth.par_chunks_mut(dst_width))
        .enumerate()
        .for_each(|(y, (row_colors, row_depths))| {
            let sy0 = y * 2;
            let sy1 = sy0 + 1;
            let row0 = sy0 * src_width;
            let row1 = sy1 * src_width;

            for x in 0..dst_width {
                let sx0 = x * 2;
                let idx0 = row0 + sx0;
                let idx1 = idx0 + 1;
                let idx2 = row1 + sx0;
                let idx3 = idx2 + 1;

                let (r0, g0, b0, a0) = src.colors[idx0];
                let (r1, g1, b1, a1) = src.colors[idx1];
                let (r2, g2, b2, a2) = src.colors[idx2];
                let (r3, g3, b3, a3) = src.colors[idx3];

                let mut r_lin = 0.0;
                let mut g_lin = 0.0;
                let mut b_lin = 0.0;
                let mut a_sum = 0.0;
                let max_z = src.depth[idx0].max(src.depth[idx1]).max(src.depth[idx2]).max(src.depth[idx3]);

                let alpha0 = a0 as f32 / 255.0;
                let alpha1 = a1 as f32 / 255.0;
                let alpha2 = a2 as f32 / 255.0;
                let alpha3 = a3 as f32 / 255.0;

                r_lin += srgb_to_linear(r0) * alpha0;
                g_lin += srgb_to_linear(g0) * alpha0;
                b_lin += srgb_to_linear(b0) * alpha0;
                a_sum += alpha0;

                r_lin += srgb_to_linear(r1) * alpha1;
                g_lin += srgb_to_linear(g1) * alpha1;
                b_lin += srgb_to_linear(b1) * alpha1;
                a_sum += alpha1;

                r_lin += srgb_to_linear(r2) * alpha2;
                g_lin += srgb_to_linear(g2) * alpha2;
                b_lin += srgb_to_linear(b2) * alpha2;
                a_sum += alpha2;

                r_lin += srgb_to_linear(r3) * alpha3;
                g_lin += srgb_to_linear(g3) * alpha3;
                b_lin += srgb_to_linear(b3) * alpha3;
                a_sum += alpha3;

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

                row_colors[x] = (r, g, b, a);
                row_depths[x] = max_z;
            }
        });

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

    let colors = &buffer.colors;
    let depth = &buffer.depth;
    let mut edge_factors = vec![0.0_f32; width * height];

    edge_factors
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            if y == 0 || y + 1 >= height {
                return;
            }

            for x in 1..width - 1 {
                let idx = y * width + x;
                if colors[idx].3 == 0 {
                    continue;
                }

                let center_depth = depth[idx];
                if center_depth <= f32::NEG_INFINITY {
                    continue;
                }

                let sample = |dx: i32, dy: i32| -> f32 {
                    let nx = (x as i32 + dx) as usize;
                    let ny = (y as i32 + dy) as usize;
                    let n_idx = ny * width + nx;
                    let n_alpha = colors[n_idx].3;
                    let n_depth = depth[n_idx];
                    if n_alpha == 0 || n_depth <= f32::NEG_INFINITY {
                        center_depth
                    } else {
                        n_depth
                    }
                };

                let d_tl = sample(-1, -1);
                let d_t = sample(0, -1);
                let d_tr = sample(1, -1);
                let d_l = sample(-1, 0);
                let d_r = sample(1, 0);
                let d_bl = sample(-1, 1);
                let d_b = sample(0, 1);
                let d_br = sample(1, 1);

                let gx = (d_tr + 2.0 * d_r + d_br) - (d_tl + 2.0 * d_l + d_bl);
                let gy = (d_bl + 2.0 * d_b + d_br) - (d_tl + 2.0 * d_t + d_tr);
                let gradient = (gx * gx + gy * gy).sqrt();

                let max_neighbor = d_tl
                    .max(d_t)
                    .max(d_tr)
                    .max(d_l)
                    .max(d_r)
                    .max(d_bl)
                    .max(d_b)
                    .max(d_br);
                let min_neighbor = d_tl
                    .min(d_t)
                    .min(d_tr)
                    .min(d_l)
                    .min(d_r)
                    .min(d_bl)
                    .min(d_b)
                    .min(d_br);
                let local_range = max_neighbor - min_neighbor;

                let edge_strength = gradient.max(local_range * 0.5);
                if edge_strength > norm_threshold {
                    let normalized = ((edge_strength - norm_threshold) / depth_range).min(1.0);
                    let factor = (normalized * strength * 2.0).min(0.7);
                    row[x] = factor;
                }
            }
        });

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

/// Fast morphological anti-aliasing based on luma contrast.
/// Smooths high-contrast edges without supersampling.
pub fn apply_edge_aa(buffer: &mut PixelBuffer, strength: f32, threshold: f32) {
    let width = buffer.width;
    let height = buffer.height;

    if width < 3 || height < 3 {
        return;
    }

    let colors = &buffer.colors;
    let mut out = buffer.colors.clone();

    // Process rows in parallel
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row_out)| {
            if y == 0 || y + 1 >= height {
                return;
            }

            let luma = |c: (u8, u8, u8, u8)| -> f32 {
                (0.299 * c.0 as f32 + 0.587 * c.1 as f32 + 0.114 * c.2 as f32) / 255.0
            };

            let row_idx = y * width;
            for x in 1..width - 1 {
                let idx = row_idx + x;
                let c = colors[idx];
                if c.3 == 0 {
                    continue;
                }

                let n = colors[idx - width];
                let s = colors[idx + width];
                let w = colors[idx - 1];
                let e = colors[idx + 1];

                let l = luma(c);
                let l_n = luma(n);
                let l_s = luma(s);
                let l_w = luma(w);
                let l_e = luma(e);

                let l_min = l.min(l_n.min(l_s).min(l_w).min(l_e));
                let l_max = l.max(l_n.max(l_s).max(l_w).max(l_e));
                let contrast = l_max - l_min;
                if contrast < threshold {
                    continue;
                }

                let horiz = (l_w - l_e).abs();
                let vert = (l_n - l_s).abs();
                let blend = (contrast * strength).min(0.8);

                let (ar, ag, ab) = if horiz >= vert {
                    (
                        ((w.0 as u16 + e.0 as u16) / 2) as u8,
                        ((w.1 as u16 + e.1 as u16) / 2) as u8,
                        ((w.2 as u16 + e.2 as u16) / 2) as u8,
                    )
                } else {
                    (
                        ((n.0 as u16 + s.0 as u16) / 2) as u8,
                        ((n.1 as u16 + s.1 as u16) / 2) as u8,
                        ((n.2 as u16 + s.2 as u16) / 2) as u8,
                    )
                };

                let inv = 1.0 - blend;
                row_out[x] = (
                    (c.0 as f32 * inv + ar as f32 * blend) as u8,
                    (c.1 as f32 * inv + ag as f32 * blend) as u8,
                    (c.2 as f32 * inv + ab as f32 * blend) as u8,
                    c.3,
                );
            }
        });

    buffer.colors = out;
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

    let colors = &buffer.colors;
    let depth = &buffer.depth;
    let mut occlusion = vec![0.0_f32; width * height];

    occlusion
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let yi = y as i32;
            for x in 0..width {
                let xi = x as i32;
                let idx = y * width + x;

                if colors[idx].3 == 0 {
                    continue;
                }

                let center_depth = depth[idx];
                if center_depth <= f32::NEG_INFINITY {
                    continue;
                }

                let mut occluded = 0_u8;
                let offsets: [(i32, i32); 4] = [(r, 0), (-r, 0), (0, r), (0, -r)];

                for (dx, dy) in offsets {
                    let sx = xi + dx;
                    let sy = yi + dy;
                    if sx >= 0 && sy >= 0 && (sx as usize) < width && (sy as usize) < height {
                        let s_idx = sy as usize * width + sx as usize;
                        if colors[s_idx].3 > 0 {
                            let sample_depth = depth[s_idx];
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
        });

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

/// Fill small gaps in the depth buffer caused by concave regions in Surface rendering.
/// This fixes "ray" artifacts where spheres don't fully overlap in surface representation.
/// Only fills pixels where neighboring opaque pixels have similar depth (within depth_eps).
pub fn fill_depth_gaps(buffer: &mut PixelBuffer, radius: usize, depth_eps: f32) {
    let width = buffer.width;
    let height = buffer.height;

    if width < radius * 2 + 1 || height < radius * 2 + 1 {
        return;
    }

    let colors = buffer.colors.clone();
    let depth = buffer.depth.clone();
    let mut out_colors = buffer.colors.clone();
    let mut out_depth = buffer.depth.clone();

    for y in radius..height - radius {
        for x in radius..width - radius {
            let idx = y * width + x;
            // Only process transparent/empty pixels
            if colors[idx].3 != 0 {
                continue;
            }

            let mut best: Option<(f32, (u8, u8, u8, u8))> = None;
            let mut min_d = f32::INFINITY;
            let mut max_d = f32::NEG_INFINITY;

            // Search neighbors within radius
            for dy in -(radius as i32)..=(radius as i32) {
                for dx in -(radius as i32)..=(radius as i32) {
                    let nx = (x as i32 + dx) as usize;
                    let ny = (y as i32 + dy) as usize;
                    let n_idx = ny * width + nx;

                    // Skip empty neighbors
                    if colors[n_idx].3 == 0 {
                        continue;
                    }

                    let d = depth[n_idx];
                    // Track depth range
                    min_d = min_d.min(d);
                    max_d = max_d.max(d);

                    // Keep track of the closest (highest z) neighbor
                    if best.is_none_or(|(bd, _)| d > bd) {
                        best = Some((d, colors[n_idx]));
                    }
                }
            }

            // Only fill if we found neighbors and their depth range is within tolerance
            // This ensures we only fill gaps between surfaces at similar depth,
            // not gaps between front and back surfaces
            if let Some((d, c)) = best {
                if max_d - min_d <= depth_eps {
                    out_colors[idx] = c;
                    // Place filled pixel slightly behind to avoid z-fighting
                    out_depth[idx] = d - 1e-3;
                }
            }
        }
    }

    buffer.colors = out_colors;
    buffer.depth = out_depth;
}
