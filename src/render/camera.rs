//! Camera and projection handling
//!
//! Implements trackball rotation and orthographic projection.

use nalgebra::{Matrix4, Quaternion, UnitQuaternion, Vector2, Vector3};

/// Camera state for viewing the molecule
#[derive(Debug, Clone)]
pub struct Camera {
    /// Current rotation as a unit quaternion
    pub rotation: UnitQuaternion<f32>,
    /// Translation (pan) in screen coordinates
    pub translation: Vector2<f32>,
    /// Zoom factor (scale)
    pub zoom: f32,
    /// Center of rotation (usually molecule center)
    pub center: Vector3<f32>,
}

impl Camera {
    /// Create a new camera centered on the given point
    pub fn new(center: Vector3<f32>) -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation: Vector2::zeros(),
            zoom: 1.0,
            center,
        }
    }

    /// Reset the camera to default orientation
    pub fn reset(&mut self, center: Vector3<f32>) {
        self.rotation = UnitQuaternion::identity();
        self.translation = Vector2::zeros();
        self.zoom = 1.0;
        self.center = center;
    }

    /// Apply trackball rotation from mouse movement
    ///
    /// `prev` and `curr` are normalized screen coordinates (-1 to 1)
    pub fn trackball_rotate(&mut self, prev: Vector2<f32>, curr: Vector2<f32>) {
        let rotation = trackball_rotation(prev, curr);
        self.rotation = rotation * self.rotation;
    }

    /// Pan the view
    pub fn pan(&mut self, delta: Vector2<f32>) {
        self.translation += delta;
    }

    /// Zoom in or out
    pub fn zoom_by(&mut self, factor: f32) {
        self.zoom *= factor;
        self.zoom = self.zoom.clamp(0.1, 100.0);
    }

    /// Transform a 3D point to screen coordinates
    pub fn project(&self, point: Vector3<f32>) -> (Vector2<f32>, f32) {
        // Translate to center of rotation
        let centered = point - self.center;

        // Apply rotation
        let rotated = self.rotation * centered;

        // Apply zoom
        let scaled = rotated * self.zoom;

        // Apply translation
        let screen_x = scaled.x + self.translation.x;
        let screen_y = scaled.y + self.translation.y;

        // Return screen position and depth (Z for depth buffer)
        (Vector2::new(screen_x, screen_y), scaled.z)
    }

    /// Fit the view to show the entire molecule
    pub fn fit_to_bounds(&mut self, min: Vector3<f32>, max: Vector3<f32>, screen_size: (u16, u16)) {
        // Calculate molecule size
        let size = max - min;
        let max_dim = size.x.max(size.y).max(size.z);

        if max_dim > 0.0 {
            // Calculate zoom to fit molecule in screen
            let screen_min = (screen_size.0 as f32).min(screen_size.1 as f32);
            self.zoom = screen_min * 0.8 / max_dim;
        }

        // Center on molecule
        self.center = (min + max) / 2.0;
        self.translation = Vector2::zeros();
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(Vector3::zeros())
    }
}

/// Calculate trackball rotation from two screen positions
///
/// Uses the arcball algorithm for intuitive rotation.
fn trackball_rotation(prev: Vector2<f32>, curr: Vector2<f32>) -> UnitQuaternion<f32> {
    const RADIUS: f32 = 0.8;

    let p1 = project_to_sphere(prev, RADIUS);
    let p2 = project_to_sphere(curr, RADIUS);

    // Rotation axis is the cross product
    let axis = p1.cross(&p2);

    // Check for valid axis
    let axis_len = axis.magnitude();
    if axis_len < 1e-6 {
        return UnitQuaternion::identity();
    }

    // Rotation angle from dot product
    let dot = p1.dot(&p2).clamp(-1.0, 1.0);
    let angle = dot.acos();

    // Create rotation quaternion
    UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle)
}

/// Project a 2D screen point onto a virtual sphere for trackball rotation
fn project_to_sphere(p: Vector2<f32>, radius: f32) -> Vector3<f32> {
    let d = p.x * p.x + p.y * p.y;
    let r2 = radius * radius;

    if d < r2 / 2.0 {
        // On the sphere
        Vector3::new(p.x, p.y, (r2 - d).sqrt())
    } else {
        // On the hyperbola (outside sphere)
        let z = r2 / 2.0 / d.sqrt();
        Vector3::new(p.x, p.y, z)
    }
}
