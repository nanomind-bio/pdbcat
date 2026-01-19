//! Bond representation

/// A chemical bond between two atoms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bond {
    /// Index of first atom in the molecule's atoms array
    pub atom1: usize,
    /// Index of second atom
    pub atom2: usize,
    /// Bond order (single, double, etc.)
    pub order: BondOrder,
}

impl Bond {
    /// Create a new bond between two atoms
    pub fn new(atom1: usize, atom2: usize, order: BondOrder) -> Self {
        Self {
            atom1,
            atom2,
            order,
        }
    }

    /// Create a single bond
    pub fn single(atom1: usize, atom2: usize) -> Self {
        Self::new(atom1, atom2, BondOrder::Single)
    }
}

/// Bond order (single, double, triple, aromatic)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BondOrder {
    #[default]
    Single,
    Double,
    Triple,
    Aromatic,
}
