//! mmCIF format parser
//!
//! Parses the following data categories:
//! - _atom_site: Atom coordinates and metadata
//! - _struct_conf: Secondary structure (helix) assignments
//! - _struct_sheet_range: Beta sheet assignments

use crate::molecule::{
    Assembly, AssemblyInstance, Atom, Element, HelixType, Molecule, SecondaryStructureAssignment,
    Transform,
};
use crate::parser::pdb::determine_bonds_shared;
use crate::parser::ParseError;
use nalgebra::{Matrix3, Vector3};
use std::collections::{HashMap, HashSet};

/// Parse mmCIF format content
pub fn parse_mmcif(content: &str) -> Result<Molecule, ParseError> {
    let mut molecule = Molecule::new();
    let mut chain_set: HashSet<char> = HashSet::new();

    // Parse _atom_site loop
    if let Some(atoms) = parse_atom_site_loop(content)? {
        for atom in atoms {
            chain_set.insert(atom.chain_id);
            molecule.atoms.push(atom);
        }
    }

    // Parse secondary structure
    molecule.secondary_structure = parse_secondary_structure(content)?;

    // Parse biological assemblies
    molecule.assemblies = parse_assemblies(content)?;

    // Collect chains
    molecule.chains = chain_set.into_iter().collect();
    molecule.chains.sort();

    // Determine bonds
    molecule.bonds = determine_bonds_shared(&molecule.atoms);

    Ok(molecule)
}

/// Parse the _atom_site loop to extract atoms
fn parse_atom_site_loop(content: &str) -> Result<Option<Vec<Atom>>, ParseError> {
    let mut atoms = Vec::new();
    let mut in_atom_site = false;
    let mut columns: Vec<String> = Vec::new();

    for line in content.lines() {
        let line = line.trim();

        // Check for start of loop
        if line.starts_with("loop_") {
            in_atom_site = false;
            columns.clear();
            continue;
        }

        if line.starts_with("_atom_site.") {
            in_atom_site = true;
            let col_name = line.strip_prefix("_atom_site.").unwrap_or("").to_string();
            columns.push(col_name);
            continue;
        }

        // End of loop or start of new category
        if in_atom_site && (line.starts_with('_') || line.starts_with('#') || line.starts_with("loop_")) {
            break;
        }

        // Parse data row
        if in_atom_site && !line.is_empty() && !line.starts_with('_') {
            if let Some(atom) = parse_atom_site_row(line, &columns)? {
                atoms.push(atom);
            }
        }
    }

    if atoms.is_empty() {
        Ok(None)
    } else {
        Ok(Some(atoms))
    }
}

/// Parse a single row of _atom_site data
fn parse_atom_site_row(line: &str, columns: &[String]) -> Result<Option<Atom>, ParseError> {
    let values: Vec<&str> = tokenize_mmcif_line(line);

    if values.len() < columns.len() {
        return Ok(None);
    }

    let col_map: HashMap<&str, &str> = columns
        .iter()
        .zip(values.iter())
        .map(|(k, v)| (k.as_str(), *v))
        .collect();

    // Only process ATOM and HETATM records, only first model
    let group_pdb = col_map.get("group_PDB").copied().unwrap_or("ATOM");
    if group_pdb != "ATOM" && group_pdb != "HETATM" {
        return Ok(None);
    }

    let model_num: i32 = col_map
        .get("pdbx_PDB_model_num")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    if model_num != 1 {
        return Ok(None);
    }

    let serial: u32 = col_map
        .get("id")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let name = col_map
        .get("label_atom_id")
        .or_else(|| col_map.get("auth_atom_id"))
        .copied()
        .unwrap_or("X")
        .to_string();

    let alt_loc = col_map.get("label_alt_id").and_then(|s| {
        let s = s.trim();
        if s == "." || s.is_empty() {
            None
        } else {
            s.chars().next()
        }
    });

    let residue_name = col_map
        .get("label_comp_id")
        .or_else(|| col_map.get("auth_comp_id"))
        .copied()
        .unwrap_or("UNK")
        .to_string();

    let chain_id = col_map
        .get("auth_asym_id")
        .or_else(|| col_map.get("label_asym_id"))
        .and_then(|s| s.chars().next())
        .unwrap_or('A');

    let residue_seq: i32 = col_map
        .get("auth_seq_id")
        .or_else(|| col_map.get("label_seq_id"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let ins_code = col_map.get("pdbx_PDB_ins_code").and_then(|s| {
        let s = s.trim();
        if s == "?" || s == "." || s.is_empty() {
            None
        } else {
            s.chars().next()
        }
    });

    let x: f32 = col_map.get("Cartn_x").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let y: f32 = col_map.get("Cartn_y").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let z: f32 = col_map.get("Cartn_z").and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let occupancy: f32 = col_map.get("occupancy").and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let temp_factor: f32 = col_map.get("B_iso_or_equiv").and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let element = col_map
        .get("type_symbol")
        .map(|s| Element::from_symbol(s))
        .unwrap_or(Element::Unknown);

    let is_hetatm = group_pdb == "HETATM";

    Ok(Some(Atom {
        serial,
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
    }))
}

/// Parse secondary structure from _struct_conf and _struct_sheet_range
fn parse_secondary_structure(content: &str) -> Result<Vec<SecondaryStructureAssignment>, ParseError> {
    let mut assignments = Vec::new();
    assignments.extend(parse_struct_conf(content)?);
    assignments.extend(parse_struct_sheet_range(content)?);
    Ok(assignments)
}

/// Parse biological assemblies from mmCIF data
fn parse_assemblies(content: &str) -> Result<Vec<Assembly>, ParseError> {
    let oper_list = parse_struct_oper_list(content)?;
    if oper_list.is_empty() {
        return Ok(Vec::new());
    }

    let gens = parse_struct_assembly_gen(content)?;
    if gens.is_empty() {
        return Ok(Vec::new());
    }

    let mut assemblies: HashMap<String, Assembly> = HashMap::new();

    for gen in gens {
        let sequences = expand_oper_expression(&gen.oper_expression);
        for seq in sequences {
            let mut combined = Transform::identity();
            let mut valid = true;
            for op_id in seq {
                if let Some(op) = oper_list.get(&op_id) {
                    combined = Transform::compose(op, &combined);
                } else {
                    valid = false;
                    break;
                }
            }

            if !valid {
                continue;
            }

            let instance = AssemblyInstance {
                transform: combined,
                chains: gen.chains.clone(),
            };

            assemblies
                .entry(gen.assembly_id.clone())
                .or_insert_with(|| Assembly {
                    id: gen.assembly_id.clone(),
                    instances: Vec::new(),
                })
                .instances
                .push(instance);
        }
    }

    let mut assemblies: Vec<Assembly> = assemblies.into_values().collect();
    assemblies.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(assemblies)
}

/// Parse helix definitions from _struct_conf
fn parse_struct_conf(content: &str) -> Result<Vec<SecondaryStructureAssignment>, ParseError> {
    let mut assignments = Vec::new();
    let mut in_struct_conf = false;
    let mut columns: Vec<String> = Vec::new();

    for line in content.lines() {
        let line = line.trim();

        if line.starts_with("loop_") {
            in_struct_conf = false;
            columns.clear();
            continue;
        }

        if line.starts_with("_struct_conf.") {
            in_struct_conf = true;
            let col_name = line.strip_prefix("_struct_conf.").unwrap_or("").to_string();
            columns.push(col_name);
            continue;
        }

        if in_struct_conf && (line.starts_with('_') || line.starts_with('#') || line.starts_with("loop_")) {
            break;
        }

        if in_struct_conf && !line.is_empty() && !line.starts_with('_') {
            let values: Vec<&str> = tokenize_mmcif_line(line);
            if values.len() >= columns.len() {
                let col_map: HashMap<&str, &str> = columns
                    .iter()
                    .zip(values.iter())
                    .map(|(k, v)| (k.as_str(), *v))
                    .collect();

                let conf_type = col_map.get("conf_type_id").copied().unwrap_or("");
                if conf_type.starts_with("HELX") {
                    let chain_id = col_map
                        .get("beg_auth_asym_id")
                        .and_then(|s| s.chars().next())
                        .unwrap_or('A');
                    let start_seq: i32 = col_map
                        .get("beg_auth_seq_id")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    let end_seq: i32 = col_map
                        .get("end_auth_seq_id")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);

                    assignments.push(SecondaryStructureAssignment::helix(
                        chain_id, start_seq, end_seq, HelixType::Alpha,
                    ));
                }
            }
        }
    }

    Ok(assignments)
}

/// Parse sheet definitions from _struct_sheet_range
fn parse_struct_sheet_range(content: &str) -> Result<Vec<SecondaryStructureAssignment>, ParseError> {
    let mut assignments = Vec::new();
    let mut in_sheet_range = false;
    let mut columns: Vec<String> = Vec::new();

    for line in content.lines() {
        let line = line.trim();

        if line.starts_with("loop_") {
            in_sheet_range = false;
            columns.clear();
            continue;
        }

        if line.starts_with("_struct_sheet_range.") {
            in_sheet_range = true;
            let col_name = line.strip_prefix("_struct_sheet_range.").unwrap_or("").to_string();
            columns.push(col_name);
            continue;
        }

        if in_sheet_range && (line.starts_with('_') || line.starts_with('#') || line.starts_with("loop_")) {
            break;
        }

        if in_sheet_range && !line.is_empty() && !line.starts_with('_') {
            let values: Vec<&str> = tokenize_mmcif_line(line);
            if values.len() >= columns.len() {
                let col_map: HashMap<&str, &str> = columns
                    .iter()
                    .zip(values.iter())
                    .map(|(k, v)| (k.as_str(), *v))
                    .collect();

                let chain_id = col_map
                    .get("beg_auth_asym_id")
                    .and_then(|s| s.chars().next())
                    .unwrap_or('A');
                let start_seq: i32 = col_map
                    .get("beg_auth_seq_id")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let end_seq: i32 = col_map
                    .get("end_auth_seq_id")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);

                assignments.push(SecondaryStructureAssignment::sheet(
                    chain_id, start_seq, end_seq,
                ));
            }
        }
    }

    Ok(assignments)
}

#[derive(Debug, Clone)]
struct AssemblyGen {
    assembly_id: String,
    oper_expression: String,
    chains: Option<Vec<char>>,
}

/// Parse assembly generator definitions
fn parse_struct_assembly_gen(content: &str) -> Result<Vec<AssemblyGen>, ParseError> {
    let mut gens = Vec::new();
    let mut in_gen = false;
    let mut columns: Vec<String> = Vec::new();

    for line in content.lines() {
        let line = line.trim();

        if line.starts_with("loop_") {
            in_gen = false;
            columns.clear();
            continue;
        }

        if line.starts_with("_pdbx_struct_assembly_gen.") {
            in_gen = true;
            let col_name = line
                .strip_prefix("_pdbx_struct_assembly_gen.")
                .unwrap_or("")
                .to_string();
            columns.push(col_name);
            continue;
        }

        if in_gen && (line.starts_with('_') || line.starts_with('#') || line.starts_with("loop_")) {
            break;
        }

        if in_gen && !line.is_empty() && !line.starts_with('_') {
            let values: Vec<&str> = tokenize_mmcif_line(line);
            if values.len() >= columns.len() {
                let col_map: HashMap<&str, &str> = columns
                    .iter()
                    .zip(values.iter())
                    .map(|(k, v)| (k.as_str(), *v))
                    .collect();

                let assembly_id = col_map
                    .get("assembly_id")
                    .copied()
                    .unwrap_or("1")
                    .to_string();
                let oper_expression = col_map
                    .get("oper_expression")
                    .copied()
                    .unwrap_or("")
                    .to_string();
                let chains = col_map
                    .get("asym_id_list")
                    .and_then(|s| parse_chain_list(s));

                gens.push(AssemblyGen {
                    assembly_id,
                    oper_expression,
                    chains,
                });
            }
        }
    }

    if gens.is_empty() {
        if let Some(gen) = parse_struct_assembly_gen_single(content)? {
            gens.push(gen);
        }
    }

    Ok(gens)
}

fn parse_struct_assembly_gen_single(content: &str) -> Result<Option<AssemblyGen>, ParseError> {
    let mut row: HashMap<String, String> = HashMap::new();
    let mut in_gen = false;

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with("_pdbx_struct_assembly_gen.") {
            in_gen = true;
            let values = tokenize_mmcif_line(line);
            if values.len() >= 2 {
                let key = values[0]
                    .strip_prefix("_pdbx_struct_assembly_gen.")
                    .unwrap_or("");
                row.insert(key.to_string(), values[1].to_string());
            }
            continue;
        }

        if in_gen && (line.starts_with('#') || line.starts_with('_') || line.starts_with("loop_")) {
            break;
        }
    }

    if row.is_empty() {
        return Ok(None);
    }

    let assembly_id = row.get("assembly_id").cloned().unwrap_or_else(|| "1".to_string());
    let oper_expression = row.get("oper_expression").cloned().unwrap_or_default();
    let chains = row.get("asym_id_list").and_then(|s| parse_chain_list(s));

    Ok(Some(AssemblyGen {
        assembly_id,
        oper_expression,
        chains,
    }))
}

/// Parse operation list definitions
fn parse_struct_oper_list(content: &str) -> Result<HashMap<String, Transform>, ParseError> {
    let mut ops = HashMap::new();
    let mut in_oper = false;
    let mut columns: Vec<String> = Vec::new();

    for line in content.lines() {
        let line = line.trim();

        if line.starts_with("loop_") {
            in_oper = false;
            columns.clear();
            continue;
        }

        if line.starts_with("_pdbx_struct_oper_list.") {
            in_oper = true;
            let col_name = line
                .strip_prefix("_pdbx_struct_oper_list.")
                .unwrap_or("")
                .to_string();
            columns.push(col_name);
            continue;
        }

        if in_oper && (line.starts_with('_') || line.starts_with('#') || line.starts_with("loop_")) {
            break;
        }

        if in_oper && !line.is_empty() && !line.starts_with('_') {
            let values: Vec<&str> = tokenize_mmcif_line(line);
            if values.len() >= columns.len() {
                let col_map: HashMap<&str, &str> = columns
                    .iter()
                    .zip(values.iter())
                    .map(|(k, v)| (k.as_str(), *v))
                    .collect();

                if let Some(id) = col_map.get("id") {
                    let transform = transform_from_lookup(|key| col_map.get(key).copied());
                    ops.insert(id.to_string(), transform);
                }
            }
        }
    }

    if ops.is_empty() {
        if let Some((id, transform)) = parse_struct_oper_list_single(content)? {
            ops.insert(id, transform);
        }
    }

    Ok(ops)
}

fn parse_struct_oper_list_single(
    content: &str,
) -> Result<Option<(String, Transform)>, ParseError> {
    let mut row: HashMap<String, String> = HashMap::new();
    let mut in_oper = false;

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with("_pdbx_struct_oper_list.") {
            in_oper = true;
            let values = tokenize_mmcif_line(line);
            if values.len() >= 2 {
                let key = values[0]
                    .strip_prefix("_pdbx_struct_oper_list.")
                    .unwrap_or("");
                row.insert(key.to_string(), values[1].to_string());
            }
            continue;
        }

        if in_oper && (line.starts_with('#') || line.starts_with('_') || line.starts_with("loop_")) {
            break;
        }
    }

    if row.is_empty() {
        return Ok(None);
    }

    let id = row.get("id").cloned().unwrap_or_else(|| "1".to_string());
    let transform = transform_from_lookup(|key| row.get(key).map(|s| s.as_str()));
    Ok(Some((id, transform)))
}

fn transform_from_lookup<'a, F>(mut lookup: F) -> Transform
where
    F: FnMut(&str) -> Option<&'a str>,
{
    let m11 = parse_float_lookup(&mut lookup, "matrix[1][1]", 1.0);
    let m12 = parse_float_lookup(&mut lookup, "matrix[1][2]", 0.0);
    let m13 = parse_float_lookup(&mut lookup, "matrix[1][3]", 0.0);
    let m21 = parse_float_lookup(&mut lookup, "matrix[2][1]", 0.0);
    let m22 = parse_float_lookup(&mut lookup, "matrix[2][2]", 1.0);
    let m23 = parse_float_lookup(&mut lookup, "matrix[2][3]", 0.0);
    let m31 = parse_float_lookup(&mut lookup, "matrix[3][1]", 0.0);
    let m32 = parse_float_lookup(&mut lookup, "matrix[3][2]", 0.0);
    let m33 = parse_float_lookup(&mut lookup, "matrix[3][3]", 1.0);

    let v1 = parse_float_lookup(&mut lookup, "vector[1]", 0.0);
    let v2 = parse_float_lookup(&mut lookup, "vector[2]", 0.0);
    let v3 = parse_float_lookup(&mut lookup, "vector[3]", 0.0);

    Transform {
        rotation: Matrix3::new(m11, m12, m13, m21, m22, m23, m31, m32, m33),
        translation: Vector3::new(v1, v2, v3),
    }
}

fn parse_float_lookup<'a, F>(lookup: &mut F, key: &str, default: f32) -> f32
where
    F: FnMut(&str) -> Option<&'a str>,
{
    lookup(key)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(default)
}

fn parse_chain_list(value: &str) -> Option<Vec<char>> {
    let value = value.trim();
    if value.is_empty() || value == "." || value == "?" {
        return None;
    }

    let mut chains = Vec::new();
    for part in value.split(',') {
        let part = part.trim();
        if let Some(ch) = part.chars().next() {
            chains.push(ch);
        }
    }

    if chains.is_empty() {
        None
    } else {
        Some(chains)
    }
}

fn expand_oper_expression(expr: &str) -> Vec<Vec<String>> {
    let expr = expr.trim();
    if expr.is_empty() || expr == "." || expr == "?" {
        return vec![Vec::new()];
    }

    let mut groups: Vec<Vec<String>> = Vec::new();

    if expr.contains('(') {
        let mut current = String::new();
        let mut in_group = false;
        for c in expr.chars() {
            if c == '(' {
                in_group = true;
                current.clear();
            } else if c == ')' {
                if in_group {
                    groups.push(parse_oper_list_expr(&current));
                }
                in_group = false;
            } else if in_group {
                current.push(c);
            }
        }
    }

    if groups.is_empty() {
        groups.push(parse_oper_list_expr(expr));
    }

    let mut sequences: Vec<Vec<String>> = vec![Vec::new()];
    for group in groups {
        if group.is_empty() {
            continue;
        }
        let mut next = Vec::new();
        for seq in &sequences {
            for op in &group {
                let mut new_seq = seq.clone();
                new_seq.push(op.clone());
                next.push(new_seq);
            }
        }
        sequences = next;
    }

    if sequences.is_empty() {
        sequences.push(Vec::new());
    }

    sequences
}

fn parse_oper_list_expr(expr: &str) -> Vec<String> {
    let mut ops = Vec::new();
    for part in expr.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some(range_ops) = parse_oper_range(part) {
            ops.extend(range_ops);
        } else {
            ops.push(part.to_string());
        }
    }
    ops
}

fn parse_oper_range(part: &str) -> Option<Vec<String>> {
    let mut iter = part.split('-');
    let start = iter.next()?;
    let end = iter.next()?;
    if iter.next().is_some() {
        return None;
    }

    let start_num: i32 = start.trim().parse().ok()?;
    let end_num: i32 = end.trim().parse().ok()?;
    let mut ops = Vec::new();

    if start_num <= end_num {
        for i in start_num..=end_num {
            ops.push(i.to_string());
        }
    } else {
        let mut i = start_num;
        while i >= end_num {
            ops.push(i.to_string());
            if i == end_num {
                break;
            }
            i -= 1;
        }
    }

    Some(ops)
}

/// Tokenize a mmCIF data line, handling quoted strings
fn tokenize_mmcif_line(line: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut chars = line.char_indices().peekable();
    let mut in_quote = false;
    let mut quote_char = '"';
    let mut token_start: Option<usize> = None;

    while let Some((i, c)) = chars.next() {
        if in_quote {
            if c == quote_char {
                if let Some(start) = token_start {
                    tokens.push(&line[start..i]);
                }
                token_start = None;
                in_quote = false;
            }
        } else if c == '"' || c == '\'' {
            quote_char = c;
            in_quote = true;
            token_start = Some(i + 1);
        } else if c.is_whitespace() {
            if let Some(start) = token_start {
                tokens.push(&line[start..i]);
                token_start = None;
            }
        } else if token_start.is_none() {
            token_start = Some(i);
        }
    }

    if let Some(start) = token_start {
        tokens.push(&line[start..]);
    }

    tokens
}
