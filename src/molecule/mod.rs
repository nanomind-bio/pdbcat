//! Molecular structure data types
//!
//! This module defines the core data structures for representing molecular
//! structures including atoms, bonds, residues, chains, and secondary structure.

mod atom;
mod assembly;
mod bond;
mod chain;
mod secondary;

pub use atom::{Atom, Element};
pub use assembly::{Assembly, AssemblyInstance, Transform};
pub use bond::{Bond, BondOrder};
pub use chain::Chain;
pub use secondary::{HelixType, SecondaryStructure, SecondaryStructureAssignment};

use nalgebra::Vector3;

/// A complete molecular structure
#[derive(Debug, Clone)]
pub struct Molecule {
    /// All atoms in the structure
    pub atoms: Vec<Atom>,
    /// All bonds between atoms
    pub bonds: Vec<Bond>,
    /// Chain identifiers present in the structure
    pub chains: Vec<char>,
    /// Secondary structure assignments
    pub secondary_structure: Vec<SecondaryStructureAssignment>,
    /// Biological assemblies
    pub assemblies: Vec<Assembly>,
    /// Model number (for NMR structures with multiple models)
    pub model: u32,
}

impl Molecule {
    /// Create a new empty molecule
    pub fn new() -> Self {
        Self {
            atoms: Vec::new(),
            bonds: Vec::new(),
            chains: Vec::new(),
            secondary_structure: Vec::new(),
            assemblies: Vec::new(),
            model: 1,
        }
    }

    /// Get the geometric center of all atoms
    pub fn center(&self) -> Vector3<f32> {
        if self.atoms.is_empty() {
            return Vector3::zeros();
        }

        let sum: Vector3<f32> = self.atoms.iter().map(|a| a.coord).sum();
        sum / self.atoms.len() as f32
    }

    /// Get the bounding box of all atoms (min, max)
    pub fn bounding_box(&self) -> (Vector3<f32>, Vector3<f32>) {
        if self.atoms.is_empty() {
            return (Vector3::zeros(), Vector3::zeros());
        }

        let mut min = self.atoms[0].coord;
        let mut max = self.atoms[0].coord;

        for atom in &self.atoms {
            min.x = min.x.min(atom.coord.x);
            min.y = min.y.min(atom.coord.y);
            min.z = min.z.min(atom.coord.z);
            max.x = max.x.max(atom.coord.x);
            max.y = max.y.max(atom.coord.y);
            max.z = max.z.max(atom.coord.z);
        }

        (min, max)
    }

    /// Get atoms for a specific chain
    pub fn atoms_in_chain(&self, chain_id: char) -> impl Iterator<Item = &Atom> {
        self.atoms.iter().filter(move |a| a.chain_id == chain_id)
    }

    /// Get backbone atoms (CA for proteins, P for nucleic acids)
    pub fn backbone_atoms(&self) -> Vec<&Atom> {
        self.atoms
            .iter()
            .filter(|a| a.name == "CA" || a.name == "P")
            .collect()
    }

    /// Get the number of unique chains
    pub fn chain_count(&self) -> usize {
        self.chains.len()
    }

    /// Get the total atom count
    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    /// Get number of biological assemblies
    pub fn assembly_count(&self) -> usize {
        if self.assemblies.is_empty() {
            1
        } else {
            self.assemblies.len()
        }
    }
}

impl Default for Molecule {
    fn default() -> Self {
        Self::new()
    }
}
