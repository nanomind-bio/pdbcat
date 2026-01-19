//! Parser regression tests

use pdbcat::parser::{parse_file, FileFormat};
use pdbcat::molecule::Element;

#[test]
fn test_element_inference_two_letter() {
    // Test that two-letter elements are correctly inferred from atom names
    let pdb_content = r#"
ATOM      1  FE  HEM A   1       0.000   0.000   0.000  1.00  0.00
ATOM      2  ZN  ZN  A   2       1.000   0.000   0.000  1.00  0.00
ATOM      3  CA  CA  A   3       2.000   0.000   0.000  1.00  0.00
ATOM      4  MG  MG  A   4       3.000   0.000   0.000  1.00  0.00
END
"#;

    let molecule = pdbcat::parser::pdb::parse_pdb(pdb_content).unwrap();

    // FE should be recognized as Iron
    assert_eq!(molecule.atoms[0].element, Element::Fe, "FE atom should be Iron");

    // ZN should be recognized as Zinc
    assert_eq!(molecule.atoms[1].element, Element::Zn, "ZN atom should be Zinc");

    // CA in residue CA should be Calcium, not Carbon
    // Note: This depends on element column being empty for inference
    assert_eq!(molecule.atoms[2].element, Element::Ca, "CA atom should be Calcium");

    // MG should be Magnesium
    assert_eq!(molecule.atoms[3].element, Element::Mg, "MG atom should be Magnesium");
}

#[test]
fn test_element_with_explicit_column() {
    // Test that explicit element column takes precedence
    let pdb_content = r#"
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
END
"#;

    let molecule = pdbcat::parser::pdb::parse_pdb(pdb_content).unwrap();

    // Even though atom name is CA, element column says C (Carbon)
    assert_eq!(molecule.atoms[0].element, Element::C, "Explicit C in element column should be Carbon");
}

#[test]
fn test_ter_record_breaks_peptide_bonds() {
    // Test that TER records prevent peptide bonds across chain breaks
    // Note: TER record format requires proper column alignment (chain at col 22, resseq at 23-26)
    let pdb_content = "
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.500   0.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.000   1.000   0.000  1.00  0.00           O
TER       5      ALA A   1
ATOM      6  N   ALA A   2       4.000   0.000   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       5.500   0.000   0.000  1.00  0.00           C
ATOM      8  C   ALA A   2       6.500   0.000   0.000  1.00  0.00           C
END
";

    let molecule = pdbcat::parser::pdb::parse_pdb(pdb_content).unwrap();

    // Check that there's no peptide bond between C of residue 1 and N of residue 2
    // (because TER record at residue 1 should break the chain)
    let has_cross_ter_bond = molecule.bonds.iter().any(|b| {
        let atom1 = &molecule.atoms[b.atom1];
        let atom2 = &molecule.atoms[b.atom2];
        (atom1.name == "C" && atom1.residue_seq == 1 && atom2.name == "N" && atom2.residue_seq == 2)
            || (atom2.name == "C" && atom2.residue_seq == 1 && atom1.name == "N" && atom1.residue_seq == 2)
    });

    assert!(!has_cross_ter_bond, "Should not have peptide bond across TER record");
}

#[test]
fn test_consecutive_residue_bonds_without_ter() {
    // Test that consecutive residues ARE bonded when there's no TER
    let pdb_content = r#"
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.500   0.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.000   1.000   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2       3.000  -1.000   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       4.500   0.000   0.000  1.00  0.00           C
ATOM      7  C   ALA A   2       5.500   0.000   0.000  1.00  0.00           C
END
"#;

    let molecule = pdbcat::parser::pdb::parse_pdb(pdb_content).unwrap();

    // Check that there IS a peptide bond between C of residue 1 and N of residue 2
    let has_peptide_bond = molecule.bonds.iter().any(|b| {
        let atom1 = &molecule.atoms[b.atom1];
        let atom2 = &molecule.atoms[b.atom2];
        (atom1.name == "C" && atom1.residue_seq == 1 && atom2.name == "N" && atom2.residue_seq == 2)
            || (atom2.name == "C" && atom2.residue_seq == 1 && atom1.name == "N" && atom1.residue_seq == 2)
    });

    assert!(has_peptide_bond, "Should have peptide bond between consecutive residues without TER");
}

#[test]
fn test_chain_break_different_chain_ids() {
    // Test that different chain IDs prevent peptide bonds
    let pdb_content = r#"
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  C   ALA A   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  N   ALA B   2       3.000   0.000   0.000  1.00  0.00           N
ATOM      4  C   ALA B   2       4.500   0.000   0.000  1.00  0.00           C
END
"#;

    let molecule = pdbcat::parser::pdb::parse_pdb(pdb_content).unwrap();

    // Check that there's no peptide bond between chains A and B
    let has_cross_chain_bond = molecule.bonds.iter().any(|b| {
        let atom1 = &molecule.atoms[b.atom1];
        let atom2 = &molecule.atoms[b.atom2];
        atom1.chain_id != atom2.chain_id
            && ((atom1.name == "C" && atom2.name == "N") || (atom1.name == "N" && atom2.name == "C"))
    });

    assert!(!has_cross_chain_bond, "Should not have peptide bond across different chains");
}

#[test]
fn test_insertion_code_handling() {
    // Test that residues with insertion codes are handled correctly
    let pdb_content = r#"
ATOM      1  N   ALA A  50       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  C   ALA A  50       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  N   ALA A  50A      3.000   0.000   0.000  1.00  0.00           N
ATOM      4  C   ALA A  50A      4.500   0.000   0.000  1.00  0.00           C
ATOM      5  N   ALA A  51       6.000   0.000   0.000  1.00  0.00           N
END
"#;

    let molecule = pdbcat::parser::pdb::parse_pdb(pdb_content).unwrap();

    // Check insertion codes were parsed
    assert!(molecule.atoms[0].ins_code.is_none(), "Residue 50 should have no insertion code");
    assert_eq!(molecule.atoms[2].ins_code, Some('A'), "Residue 50A should have insertion code A");
}

#[test]
fn test_basic_pdb_parsing() {
    // Test basic PDB parsing with example file
    let pdb_content = include_str!("../examples/1UBQ.pdb");
    let molecule = pdbcat::parser::pdb::parse_pdb(pdb_content).unwrap();

    assert!(!molecule.atoms.is_empty(), "Should have parsed atoms");
    assert!(!molecule.chains.is_empty(), "Should have parsed chains");
}
