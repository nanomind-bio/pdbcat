//! File format parsers for PDB and mmCIF files

pub mod pdb;
pub mod mmcif;
mod topology;

pub use pdb::parse_pdb;
pub use pdb::determine_bonds_shared;
pub use mmcif::parse_mmcif;

use crate::molecule::Molecule;
use std::path::Path;
use thiserror::Error;

/// Supported file formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    /// Protein Data Bank format
    Pdb,
    /// Macromolecular Crystallographic Information File format
    MmCif,
}

/// Errors that can occur during parsing
#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Failed to read file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Parse error at line {line}: {message}")]
    ParseError { line: usize, message: String },

    #[error("Invalid coordinate value: {0}")]
    InvalidCoordinate(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid file format: {0}")]
    InvalidFormat(String),
}

/// Parse a molecular structure file
pub fn parse_file(path: &Path, format: FileFormat) -> Result<Molecule, ParseError> {
    let content = std::fs::read_to_string(path)?;

    match format {
        FileFormat::Pdb => parse_pdb(&content),
        FileFormat::MmCif => parse_mmcif(&content),
    }
}
