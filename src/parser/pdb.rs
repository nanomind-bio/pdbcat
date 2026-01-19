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
    // Track TER record positions to mark chain breaks
    let mut ter_positions: HashSet<(char, i32)> = HashSet::new();

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
            "TER" if in_first_model => {
                // Parse TER record to mark chain termination
                // TER records mark the end of a polymer chain
                // Handle both standard (long) and short TER records
                let padded = format!("{:80}", line);
                let chain_id = padded.chars().nth(21).unwrap_or(' ');
                let res_seq: i32 = padded[22..26].trim().parse().unwrap_or(0);
                if chain_id != ' ' || res_seq != 0 {
                    ter_positions.insert((chain_id, res_seq));
                }
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
    molecule.bonds = determine_bonds_with_ter(&molecule.atoms, &ter_positions);

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
        // Infer element from atom name when element column is empty
        infer_element_from_name(&name)
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

/// Infer element from atom name when element column is empty
/// Tries two-letter elements first (e.g., "FE", "CA", "ZN"), then single letter
fn infer_element_from_name(name: &str) -> Element {
    // Extract only alphabetic characters from the name
    let letters: String = name.chars().filter(|c| c.is_ascii_alphabetic()).collect();

    if letters.is_empty() {
        return Element::Unknown;
    }

    // Try two-letter element first (handles Fe, Zn, Ca, Mg, Na, Cl, Br, Se, etc.)
    if letters.len() >= 2 {
        let two_letter = &letters[0..2];
        let elem = Element::from_symbol(two_letter);
        if elem != Element::Unknown {
            return elem;
        }
    }

    // Fall back to single letter
    Element::from_symbol(&letters[0..1])
}

/// Determine bonds using residue topology and distance heuristics
/// This function is public so it can be shared with the mmCIF parser
pub fn determine_bonds_shared(atoms: &[Atom]) -> Vec<Bond> {
    // Call with empty TER positions for backward compatibility
    determine_bonds_with_ter(atoms, &HashSet::new())
}

/// Determine bonds with TER record awareness
/// TER records mark chain terminations and prevent peptide bonds across breaks
fn determine_bonds_with_ter(atoms: &[Atom], ter_positions: &HashSet<(char, i32)>) -> Vec<Bond> {
    let mut bonds = Vec::new();

    // Group atoms by residue (including insertion code for uniqueness)
    let mut residue_atoms: HashMap<(char, i32, Option<char>), Vec<usize>> = HashMap::new();
    for (idx, atom) in atoms.iter().enumerate() {
        residue_atoms
            .entry((atom.chain_id, atom.residue_seq, atom.ins_code))
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
    // Respects TER records and insertion codes
    let mut prev_c: Option<usize> = None;
    let mut prev_chain: Option<char> = None;
    let mut prev_seq: Option<i32> = None;
    let mut prev_ins_code: Option<char> = None;

    for (idx, atom) in atoms.iter().enumerate() {
        if atom.name == "N" && !atom.is_hetatm {
            if let (Some(c_idx), Some(pc), Some(ps)) = (prev_c, prev_chain, prev_seq) {
                // Check if this is the same chain
                if atom.chain_id == pc {
                    // Check for TER record between previous C and current N
                    let has_ter = ter_positions.contains(&(pc, ps));

                    if !has_ter {
                        // Determine if residues are consecutive
                        // Handle insertion codes: residues with same seq number but different
                        // insertion codes (e.g., 50, 50A, 50B) are consecutive
                        let is_consecutive = if atom.residue_seq == ps + 1 && atom.ins_code.is_none() && prev_ins_code.is_none() {
                            // Simple case: sequential residue numbers, no insertion codes
                            true
                        } else if atom.residue_seq == ps {
                            // Same residue number - check insertion code progression
                            // e.g., 50 -> 50A, or 50A -> 50B
                            match (prev_ins_code, atom.ins_code) {
                                (None, Some(_)) => true,  // 50 -> 50A
                                (Some(p), Some(c)) => (c as u8) == (p as u8) + 1,  // 50A -> 50B
                                _ => false,
                            }
                        } else if atom.residue_seq == ps + 1 && prev_ins_code.is_some() && atom.ins_code.is_none() {
                            // e.g., 50B -> 51
                            true
                        } else {
                            false
                        };

                        if is_consecutive {
                            bonds.push(Bond::single(c_idx, idx));
                        }
                    }
                }
            }
        }
        if atom.name == "C" && !atom.is_hetatm {
            prev_c = Some(idx);
            prev_chain = Some(atom.chain_id);
            prev_seq = Some(atom.residue_seq);
            prev_ins_code = atom.ins_code;
        }
    }

    // For HETATMs and unknown residues, use distance-based bonding
    add_distance_bonds(atoms, &mut bonds);

    bonds
}

/// Add bonds based on distance for atoms without topology
/// Uses spatial hashing for O(n) average complexity instead of O(n²)
fn add_distance_bonds(atoms: &[Atom], bonds: &mut Vec<Bond>) {
    let hetatm_indices: Vec<usize> = atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| a.is_hetatm && !a.is_water())
        .map(|(i, _)| i)
        .collect();

    if hetatm_indices.is_empty() {
        return;
    }

    // Use spatial hashing with cell size based on max possible bond distance
    // Max VdW radius is ~2.75 (K), so max bond dist is ~2.75 + 2.75 - 0.4 = 5.1
    const CELL_SIZE: f32 = 5.5;

    // Build spatial hash grid
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for &idx in &hetatm_indices {
        let coord = &atoms[idx].coord;
        let cell = (
            (coord.x / CELL_SIZE).floor() as i32,
            (coord.y / CELL_SIZE).floor() as i32,
            (coord.z / CELL_SIZE).floor() as i32,
        );
        grid.entry(cell).or_default().push(idx);
    }

    // Check for bonds using spatial neighbors
    let mut checked: HashSet<(usize, usize)> = HashSet::new();

    for &idx1 in &hetatm_indices {
        let coord1 = &atoms[idx1].coord;
        let cell = (
            (coord1.x / CELL_SIZE).floor() as i32,
            (coord1.y / CELL_SIZE).floor() as i32,
            (coord1.z / CELL_SIZE).floor() as i32,
        );

        // Check this cell and all 26 neighboring cells
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbor_cell = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                    if let Some(neighbors) = grid.get(&neighbor_cell) {
                        for &idx2 in neighbors {
                            if idx1 >= idx2 {
                                continue; // Avoid duplicates
                            }

                            let pair = (idx1.min(idx2), idx1.max(idx2));
                            if checked.contains(&pair) {
                                continue;
                            }
                            checked.insert(pair);

                            let dist = (atoms[idx1].coord - atoms[idx2].coord).magnitude();
                            let max_bond_dist =
                                atoms[idx1].vdw_radius() + atoms[idx2].vdw_radius() - 0.4;

                            if dist < max_bond_dist && dist > 0.4 {
                                bonds.push(Bond::single(idx1, idx2));
                            }
                        }
                    }
                }
            }
        }
    }
}
