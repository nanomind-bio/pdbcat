//! pdbcat - Terminal-based PDB/mmCIF molecular structure viewer
//!
//! A fast, keyboard-driven viewer for molecular structure files using braille
//! unicode characters for maximum resolution within terminal constraints.

use clap::Parser;
use std::path::PathBuf;
use std::process::ExitCode;

mod molecule;
mod parser;
mod render;
mod ui;

/// Terminal-based PDB/mmCIF molecular structure viewer
#[derive(Parser, Debug)]
#[command(name = "pdbcat")]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to PDB or mmCIF file to view
    #[arg(value_name = "FILE")]
    file: PathBuf,
}

fn main() -> ExitCode {
    let args = Args::parse();

    // Validate file exists
    if !args.file.exists() {
        eprintln!("Error: File not found: {}", args.file.display());
        return ExitCode::from(1);
    }

    // Validate file extension
    let extension = args
        .file
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    let file_format = match extension.as_deref() {
        Some("pdb") => parser::FileFormat::Pdb,
        Some("cif") | Some("mmcif") => parser::FileFormat::MmCif,
        _ => {
            eprintln!(
                "Error: Invalid file format for {}. Expected: .pdb, .cif, or .mmcif",
                args.file.display()
            );
            return ExitCode::from(1);
        }
    };

    // Parse the molecular structure file
    let molecule = match parser::parse_file(&args.file, file_format) {
        Ok(mol) => mol,
        Err(e) => {
            eprintln!("Error: {}", e);
            return ExitCode::from(1);
        }
    };

    // Run the interactive viewer
    if let Err(e) = ui::run(&args.file, molecule) {
        eprintln!("Error: {}", e);
        return ExitCode::from(1);
    }

    ExitCode::SUCCESS
}
