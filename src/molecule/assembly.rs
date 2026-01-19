//! Biological assembly representations

use nalgebra::{Matrix3, Vector3};

/// Rigid transform for assembly operations
#[derive(Debug, Clone)]
pub struct Transform {
    pub rotation: Matrix3<f32>,
    pub translation: Vector3<f32>,
}

impl Transform {
    pub fn identity() -> Self {
        Self {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        }
    }

    pub fn apply(&self, point: Vector3<f32>) -> Vector3<f32> {
        self.rotation * point + self.translation
    }

    /// Compose transforms: apply `before`, then `after`.
    pub fn compose(after: &Transform, before: &Transform) -> Transform {
        Transform {
            rotation: after.rotation * before.rotation,
            translation: after.rotation * before.translation + after.translation,
        }
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

/// One instance of an assembly (transform + optional chain filter)
#[derive(Debug, Clone)]
pub struct AssemblyInstance {
    pub transform: Transform,
    pub chains: Option<Vec<char>>,
}

impl AssemblyInstance {
    pub fn applies_to_chain(&self, chain_id: char) -> bool {
        match &self.chains {
            Some(chains) => chains.contains(&chain_id),
            None => true,
        }
    }
}

/// Biological assembly definition
#[derive(Debug, Clone)]
pub struct Assembly {
    pub id: String,
    pub instances: Vec<AssemblyInstance>,
}

impl Assembly {
    pub fn single_identity() -> Self {
        Self {
            id: "1".to_string(),
            instances: vec![AssemblyInstance {
                transform: Transform::identity(),
                chains: None,
            }],
        }
    }
}
