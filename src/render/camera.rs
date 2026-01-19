//! Camera and projection handling
//!
//! Implements trackball rotation with orthographic and perspective projection.

use nalgebra::{UnitQuaternion, Vector2, Vector3};

/// Projection mode for the camera
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProjectionMode {
    /// Orthographic projection (no perspective distortion)
    #[default]
    Orthographic,
    /// Perspective projection (distant objects appear smaller)
    Perspective,
}

impl ProjectionMode {
    /// Toggle between projection modes
    pub fn toggle(self) -> Self {
        match self {
            ProjectionMode::Orthographic => ProjectionMode::Perspective,
            ProjectionMode::Perspective => ProjectionMode::Orthographic,
        }
    }

    /// Get the display name
    pub fn name(&self) -> &'static str {
        match self {
            ProjectionMode::Orthographic => "Ortho",
            ProjectionMode::Perspective => "Persp",
        }
    }
}

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
    /// Projection mode (orthographic or perspective)
    pub projection: ProjectionMode,
    /// Field of view for perspective projection (in radians)
    pub fov: f32,
}

impl Camera {
    /// Create a new camera centered on the given point
    pub fn new(center: Vector3<f32>) -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation: Vector2::zeros(),
            zoom: 1.0,
            center,
            projection: ProjectionMode::default(),
            fov: std::f32::consts::FRAC_PI_4, // 45 degrees
        }
    }

    /// Reset the camera to default orientation
    pub fn reset(&mut self, center: Vector3<f32>) {
        self.rotation = UnitQuaternion::identity();
        self.translation = Vector2::zeros();
        self.zoom = 1.0;
        self.center = center;
        // Keep projection mode on reset
    }

    /// Toggle between orthographic and perspective projection
    pub fn toggle_projection(&mut self) {
        self.projection = self.projection.toggle();
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
        // Allow small zoom values for large molecules (max_dim ~100Å gives zoom ~0.02)
        self.zoom = self.zoom.clamp(0.001, 100.0);
    }

    /// Transform a 3D point to screen coordinates
    ///
    /// For perspective projection, the `screen_scale` parameter determines
    /// the distance from camera to the projection plane.
    pub fn project(&self, point: Vector3<f32>) -> (Vector2<f32>, f32) {
        self.project_with_scale(point, 1.0)
    }

    /// Transform a 3D point to screen coordinates with a screen scale factor.
    /// This is used for perspective projection to control the viewing distance.
    pub fn project_with_scale(&self, point: Vector3<f32>, screen_scale: f32) -> (Vector2<f32>, f32) {
        // Translate to center of rotation
        let centered = point - self.center;

        // Apply rotation
        let rotated = self.rotation * centered;

        // Apply zoom
        let scaled = rotated * self.zoom;

        match self.projection {
            ProjectionMode::Orthographic => {
                // Apply translation
                let screen_x = scaled.x + self.translation.x;
                let screen_y = scaled.y + self.translation.y;
                (Vector2::new(screen_x, screen_y), scaled.z)
            }
            ProjectionMode::Perspective => {
                // Perspective division
                // view_distance is the distance from camera to the projection plane
                let view_distance = screen_scale * 2.0; // Base viewing distance

                // In our coordinate system, positive z is closer to camera
                // For perspective, closer objects should appear larger
                // We compute distance from camera as (view_distance - scaled.z)
                let z_from_camera = (view_distance - scaled.z).max(0.1);

                // Perspective divide - closer objects (smaller z_from_camera) appear larger
                let perspective_factor = view_distance / z_from_camera;
                let screen_x = scaled.x * perspective_factor + self.translation.x;
                let screen_y = scaled.y * perspective_factor + self.translation.y;

                // Return screen position and depth
                (Vector2::new(screen_x, screen_y), scaled.z)
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection_toggle() {
        let mut mode = ProjectionMode::Orthographic;
        mode = mode.toggle();
        assert_eq!(mode, ProjectionMode::Perspective);
        mode = mode.toggle();
        assert_eq!(mode, ProjectionMode::Orthographic);
    }

    #[test]
    fn test_perspective_depth_ordering() {
        let mut camera = Camera::new(Vector3::zeros());
        camera.projection = ProjectionMode::Perspective;
        camera.zoom = 10.0;

        // Point closer to camera should have higher screen coordinates
        let near_point = Vector3::new(10.0, 0.0, 5.0);
        let far_point = Vector3::new(10.0, 0.0, -5.0);

        let (near_screen, _) = camera.project_with_scale(near_point, 100.0);
        let (far_screen, _) = camera.project_with_scale(far_point, 100.0);

        // Near point should appear larger (further from center)
        assert!(near_screen.x.abs() > far_screen.x.abs());
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
