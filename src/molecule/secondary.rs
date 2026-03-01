//! Secondary structure definitions

/// Type of secondary structure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum SecondaryStructure {
    /// Alpha helix, 3-10 helix, or pi helix
    Helix(HelixType),
    /// Beta sheet
    Sheet,
    /// Random coil or unstructured
    #[default]
    Coil,
}

/// Types of helical secondary structure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum HelixType {
    /// Right-handed alpha helix (most common)
    #[default]
    Alpha,
    /// 3-10 helix
    ThreeTen,
    /// Pi helix
    Pi,
}

/// Assignment of secondary structure to a range of residues
#[derive(Debug, Clone)]
pub struct SecondaryStructureAssignment {
    /// Chain identifier
    pub chain_id: char,
    /// Starting residue sequence number
    pub start_seq: i32,
    /// Ending residue sequence number
    pub end_seq: i32,
    /// Type of secondary structure
    pub ss_type: SecondaryStructure,
}

impl SecondaryStructureAssignment {
    /// Create a new helix assignment
    pub fn helix(chain_id: char, start_seq: i32, end_seq: i32, helix_type: HelixType) -> Self {
        Self {
            chain_id,
            start_seq,
            end_seq,
            ss_type: SecondaryStructure::Helix(helix_type),
        }
    }

    /// Create a new sheet assignment
    pub fn sheet(chain_id: char, start_seq: i32, end_seq: i32) -> Self {
        Self {
            chain_id,
            start_seq,
            end_seq,
            ss_type: SecondaryStructure::Sheet,
        }
    }

    /// Check if a residue falls within this assignment
    pub fn contains(&self, chain_id: char, residue_seq: i32) -> bool {
        self.chain_id == chain_id && residue_seq >= self.start_seq && residue_seq <= self.end_seq
    }
}
