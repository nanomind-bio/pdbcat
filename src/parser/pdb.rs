//! PDB format parser
//!
//! Parses the following record types:
//! - ATOM: Standard amino acid atoms
//! - HETATM: Heteroatoms (ligands, waters, ions)
//! - HELIX: Alpha helix definitions
//! - SHEET: Beta sheet definitions
//! - MODEL/ENDMDL: NMR model boundaries (only first model is loaded)
//! - TER: Chain termination
//! - END: File end

use crate::molecule::{
    Atom, Bond, Element, HelixType, Molecule, SecondaryStructureAssignment,
};
use crate::parser::topology;
use crate::parser::ParseError;
use nalgebra::Vector3;
use std::collections::{HashMap, HashSet};

/// Parse PDB format content
pub fn parse_pdb(content: &str) -> Result<Molecule, ParseError> {
    let mut molecule = Molecule::new();
    let mut in_first_model = true;
    let mut seen_model = false;
    let mut chain_set: HashSet<char> = HashSet::new();

    for (line_num, line) in content.lines().enumerate() {
        let record_type = if line.len() >= 6 {
            &line[0..6]
        } else {
            line
        };

        match record_type.trim() {
            "MODEL" => {
                if seen_model {
                    in_first_model = false;
                }
                seen_model = true;
            }
            "ENDMDL" => {
                if seen_model {
                    break;
                }
            }
            "ATOM" | "HETATM" if in_first_model => {
                let atom = parse_atom_record(line, line_num + 1, record_type.trim() == "HETATM")?;
                chain_set.insert(atom.chain_id);
                molecule.atoms.push(atom);
            }
            "HELIX" => {
                if let Some(ss) = parse_helix_record(line, line_num + 1)? {
                    molecule.secondary_structure.push(ss);
                }
            }
            "SHEET" => {
                if let Some(ss) = parse_sheet_record(line, line_num + 1)? {
                    molecule.secondary_structure.push(ss);
                }
            }
            "END" => break,
            _ => {}
        }
    }

    molecule.chains = chain_set.into_iter().collect();
    molecule.chains.sort();
    molecule.bonds = determine_bonds_shared(&molecule.atoms);

    Ok(molecule)
}

/// Parse an ATOM or HETATM record
fn parse_atom_record(line: &str, line_num: usize, is_hetatm: bool) -> Result<Atom, ParseError> {
    let line = format!("{:80}", line);

    let serial = parse_int(&line[6..11], line_num, "serial number")?;
    let name = line[12..16].trim().to_string();
    let alt_loc = {
        let c = line.chars().nth(16).unwrap_or(' ');
        if c == ' ' { None } else { Some(c) }
    };
    let residue_name = line[17..20].trim().to_string();
    let chain_id = line.chars().nth(21).unwrap_or('A');
    let residue_seq = parse_int(&line[22..26], line_num, "residue sequence")?;
    let ins_code = {
        let c = line.chars().nth(26).unwrap_or(' ');
        if c == ' ' { None } else { Some(c) }
    };

    let x = parse_float(&line[30..38], line_num, "X coordinate")?;
    let y = parse_float(&line[38..46], line_num, "Y coordinate")?;
    let z = parse_float(&line[46..54], line_num, "Z coordinate")?;

    let occupancy = parse_float(&line[54..60], line_num, "occupancy").unwrap_or(1.0);
    let temp_factor = parse_float(&line[60..66], line_num, "temperature factor").unwrap_or(0.0);

    let element_str = line[76..78].trim();
    let element = if !element_str.is_empty() {
        Element::from_symbol(element_str)
    } else {
        let name_chars: String = name.chars().filter(|c| c.is_alphabetic()).collect();
        if name_chars.is_empty() {
            Element::Unknown
        } else {
            Element::from_symbol(&name_chars[0..1])
        }
    };

    Ok(Atom {
        serial: serial as u32,
        name,
        alt_loc,
        residue_name,
        chain_id,
        residue_seq,
        ins_code,
        coord: Vector3::new(x, y, z),
        occupancy,
        temp_factor,
        element,
        is_hetatm,
    })
}

/// Parse a HELIX record
fn parse_helix_record(
    line: &str,
    _line_num: usize,
) -> Result<Option<SecondaryStructureAssignment>, ParseError> {
    let line = format!("{:80}", line);

    let chain_id = line.chars().nth(19).unwrap_or(' ');
    let start_seq: i32 = line[21..25].trim().parse().unwrap_or(0);
    let end_seq: i32 = line[33..37].trim().parse().unwrap_or(0);
    let helix_class: i32 = line[38..40].trim().parse().unwrap_or(1);

    let helix_type = match helix_class {
        1 => HelixType::Alpha,
        3 => HelixType::Pi,
        5 => HelixType::ThreeTen,
        _ => HelixType::Alpha,
    };

    Ok(Some(SecondaryStructureAssignment::helix(
        chain_id, start_seq, end_seq, helix_type,
    )))
}

/// Parse a SHEET record
fn parse_sheet_record(
    line: &str,
    _line_num: usize,
) -> Result<Option<SecondaryStructureAssignment>, ParseError> {
    let line = format!("{:80}", line);

    let chain_id = line.chars().nth(21).unwrap_or(' ');
    let start_seq: i32 = line[22..26].trim().parse().unwrap_or(0);
    let end_seq: i32 = line[33..37].trim().parse().unwrap_or(0);

    Ok(Some(SecondaryStructureAssignment::sheet(
        chain_id, start_seq, end_seq,
    )))
}

fn parse_int(s: &str, line_num: usize, field: &str) -> Result<i32, ParseError> {
    s.trim()
        .parse()
        .map_err(|_| ParseError::ParseError {
            line: line_num,
            message: format!("Invalid {} value: '{}'", field, s.trim()),
        })
}

fn parse_float(s: &str, line_num: usize, field: &str) -> Result<f32, ParseError> {
    s.trim()
        .parse()
        .map_err(|_| ParseError::ParseError {
            line: line_num,
            message: format!("Invalid {} value: '{}'", field, s.trim()),
        })
}

/// Determine bonds using residue topology and distance heuristics
/// This function is public so it can be shared with the mmCIF parser
pub fn determine_bonds_shared(atoms: &[Atom]) -> Vec<Bond> {
    let mut bonds = Vec::new();

    // Group atoms by residue
    let mut residue_atoms: HashMap<(char, i32), Vec<usize>> = HashMap::new();
    for (idx, atom) in atoms.iter().enumerate() {
        residue_atoms
            .entry((atom.chain_id, atom.residue_seq))
            .or_default()
            .push(idx);
    }

    // Add intra-residue bonds from topology
    for (_key, indices) in &residue_atoms {
        if indices.is_empty() {
            continue;
        }
        let residue_name = &atoms[indices[0]].residue_name;
        let topo_bonds = topology::get_residue_bonds(residue_name);

        for (name1, name2) in topo_bonds {
            let atom1_idx = indices.iter().find(|&&i| atoms[i].name == name1).copied();
            let atom2_idx = indices.iter().find(|&&i| atoms[i].name == name2).copied();

            if let (Some(a1), Some(a2)) = (atom1_idx, atom2_idx) {
                bonds.push(Bond::single(a1, a2));
            }
        }
    }

    // Add peptide bonds (C-N between consecutive residues)
    let mut prev_c: Option<usize> = None;
    let mut prev_chain: Option<char> = None;
    let mut prev_seq: Option<i32> = None;

    for (idx, atom) in atoms.iter().enumerate() {
        if atom.name == "N" {
            if let (Some(c_idx), Some(pc), Some(ps)) = (prev_c, prev_chain, prev_seq) {
                if atom.chain_id == pc && atom.residue_seq == ps + 1 {
                    bonds.push(Bond::single(c_idx, idx));
                }
            }
        }
        if atom.name == "C" && !atom.is_hetatm {
            prev_c = Some(idx);
            prev_chain = Some(atom.chain_id);
            prev_seq = Some(atom.residue_seq);
        }
    }

    // For HETATMs and unknown residues, use distance-based bonding
    add_distance_bonds(atoms, &mut bonds);

    bonds
}

/// Add bonds based on distance for atoms without topology
fn add_distance_bonds(atoms: &[Atom], bonds: &mut Vec<Bond>) {
    let hetatm_indices: Vec<usize> = atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| a.is_hetatm && !a.is_water())
        .map(|(i, _)| i)
        .collect();

    for (i, &idx1) in hetatm_indices.iter().enumerate() {
        for &idx2 in hetatm_indices.iter().skip(i + 1) {
            let dist = (atoms[idx1].coord - atoms[idx2].coord).magnitude();
            let max_bond_dist = atoms[idx1].vdw_radius() + atoms[idx2].vdw_radius() - 0.4;

            if dist < max_bond_dist && dist > 0.4 {
                bonds.push(Bond::single(idx1, idx2));
            }
        }
    }
}
