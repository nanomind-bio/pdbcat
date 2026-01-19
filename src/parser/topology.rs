//! Residue topology library for standard amino acids and nucleotides
//!
//! Defines the expected bonds within each residue type.

/// Get the bonds for a standard residue
///
/// Returns a list of (atom_name1, atom_name2) pairs for intra-residue bonds
pub fn get_residue_bonds(residue_name: &str) -> Vec<(&'static str, &'static str)> {
    match residue_name {
        // Standard amino acids - backbone
        "ALA" | "ARG" | "ASN" | "ASP" | "CYS" | "GLN" | "GLU" | "GLY" | "HIS" | "ILE" | "LEU"
        | "LYS" | "MET" | "PHE" | "PRO" | "SER" | "THR" | "TRP" | "TYR" | "VAL" => {
            let mut bonds = backbone_bonds();
            bonds.extend(sidechain_bonds(residue_name));
            bonds
        }
        // Nucleotides
        "A" | "G" | "C" | "U" | "DA" | "DG" | "DC" | "DT" | "ADE" | "GUA" | "CYT" | "URA" | "THY" => {
            nucleotide_bonds()
        }
        // Unknown residue
        _ => Vec::new(),
    }
}

/// Standard protein backbone bonds
fn backbone_bonds() -> Vec<(&'static str, &'static str)> {
    vec![
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"), // Most amino acids except Gly
    ]
}

/// Sidechain-specific bonds for each amino acid
fn sidechain_bonds(residue_name: &str) -> Vec<(&'static str, &'static str)> {
    match residue_name {
        "ALA" => vec![],
        "ARG" => vec![
            ("CB", "CG"),
            ("CG", "CD"),
            ("CD", "NE"),
            ("NE", "CZ"),
            ("CZ", "NH1"),
            ("CZ", "NH2"),
        ],
        "ASN" => vec![("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")],
        "ASP" => vec![("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")],
        "CYS" => vec![("CB", "SG")],
        "GLN" => vec![
            ("CB", "CG"),
            ("CG", "CD"),
            ("CD", "OE1"),
            ("CD", "NE2"),
        ],
        "GLU" => vec![
            ("CB", "CG"),
            ("CG", "CD"),
            ("CD", "OE1"),
            ("CD", "OE2"),
        ],
        "GLY" => vec![], // No CB
        "HIS" => vec![
            ("CB", "CG"),
            ("CG", "ND1"),
            ("CG", "CD2"),
            ("ND1", "CE1"),
            ("CD2", "NE2"),
            ("CE1", "NE2"),
        ],
        "ILE" => vec![
            ("CB", "CG1"),
            ("CB", "CG2"),
            ("CG1", "CD1"),
        ],
        "LEU" => vec![("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
        "LYS" => vec![
            ("CB", "CG"),
            ("CG", "CD"),
            ("CD", "CE"),
            ("CE", "NZ"),
        ],
        "MET" => vec![("CB", "CG"), ("CG", "SD"), ("SD", "CE")],
        "PHE" => vec![
            ("CB", "CG"),
            ("CG", "CD1"),
            ("CG", "CD2"),
            ("CD1", "CE1"),
            ("CD2", "CE2"),
            ("CE1", "CZ"),
            ("CE2", "CZ"),
        ],
        "PRO" => vec![("CB", "CG"), ("CG", "CD"), ("CD", "N")],
        "SER" => vec![("CB", "OG")],
        "THR" => vec![("CB", "OG1"), ("CB", "CG2")],
        "TRP" => vec![
            ("CB", "CG"),
            ("CG", "CD1"),
            ("CG", "CD2"),
            ("CD1", "NE1"),
            ("CD2", "CE2"),
            ("CD2", "CE3"),
            ("NE1", "CE2"),
            ("CE2", "CZ2"),
            ("CE3", "CZ3"),
            ("CZ2", "CH2"),
            ("CZ3", "CH2"),
        ],
        "TYR" => vec![
            ("CB", "CG"),
            ("CG", "CD1"),
            ("CG", "CD2"),
            ("CD1", "CE1"),
            ("CD2", "CE2"),
            ("CE1", "CZ"),
            ("CE2", "CZ"),
            ("CZ", "OH"),
        ],
        "VAL" => vec![("CB", "CG1"), ("CB", "CG2")],
        _ => vec![],
    }
}

/// Standard nucleotide backbone and base bonds
fn nucleotide_bonds() -> Vec<(&'static str, &'static str)> {
    vec![
        // Backbone
        ("P", "O5'"),
        ("O5'", "C5'"),
        ("C5'", "C4'"),
        ("C4'", "C3'"),
        ("C3'", "O3'"),
        ("C4'", "O4'"),
        ("O4'", "C1'"),
        ("C1'", "C2'"),
        ("C2'", "C3'"),
        // Base attachment
        ("C1'", "N9"),  // Purines
        ("C1'", "N1"),  // Pyrimidines
    ]
}
