//! Braille character rendering utilities
//!
//! Provides depth cueing for shading effects.

/// Apply depth cueing to a color
/// Makes objects farther from the camera darker for depth perception
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
