//! Chain representation

/// A polypeptide or polynucleotide chain
#[derive(Debug, Clone)]
pub struct Chain {
    /// Single-character chain identifier (A, B, C, etc.)
    pub id: char,
    /// Whether this chain is currently visible
    pub visible: bool,
}

impl Chain {
    /// Create a new chain
    pub fn new(id: char) -> Self {
        Self { id, visible: true }
    }
}
