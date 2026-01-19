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

    /// Run benchmark mode: rotate for 2 seconds with high quality shading,
    /// then output performance metrics to benchmark.log
    #[arg(long, short = 'b')]
    benchmark: bool,

    /// Output a PNG image instead of running interactive viewer
    #[arg(long, short = 'o', value_name = "OUTPUT")]
    output: Option<PathBuf>,

    /// Resolution for PNG output (WxH format, e.g., 800x600)
    #[arg(long, short = 'r', default_value = "800x600")]
    resolution: String,

    /// Representation mode for PNG output (cartoon, ball-and-stick, surface, backbone)
    #[arg(long, default_value = "cartoon")]
    repr: String,
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

    // Run the viewer or output PNG
    let result = if let Some(output_path) = args.output {
        // Parse resolution
        let (width, height) = match parse_resolution(&args.resolution) {
            Some(res) => res,
            None => {
                eprintln!("Error: Invalid resolution format '{}'. Expected WxH (e.g., 800x600)", args.resolution);
                return ExitCode::from(1);
            }
        };

        // Parse representation
        let repr = match args.repr.to_lowercase().as_str() {
            "cartoon" => render::Representation::Cartoon,
            "ball-and-stick" | "ballandstick" | "bas" => render::Representation::BallAndStick,
            "surface" => render::Representation::Surface,
            "backbone" => render::Representation::Backbone,
            _ => {
                eprintln!("Error: Unknown representation '{}'. Use: cartoon, ball-and-stick, surface, backbone", args.repr);
                return ExitCode::from(1);
            }
        };

        ui::render_to_png(&args.file, molecule, &output_path, width, height, repr)
    } else if args.benchmark {
        ui::run_benchmark(&args.file, molecule)
    } else {
        ui::run(&args.file, molecule)
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        return ExitCode::from(1);
    }

    ExitCode::SUCCESS
}

fn parse_resolution(s: &str) -> Option<(usize, usize)> {
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() != 2 {
        return None;
    }
    let width = parts[0].parse().ok()?;
    let height = parts[1].parse().ok()?;
    if width == 0 || height == 0 {
        return None;
    }
    Some((width, height))
}
