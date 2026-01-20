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

    /// Interactive mode with keyboard controls
    #[arg(long, short = 'i')]
    interactive: bool,

    /// Run benchmark mode: rotate for 2 seconds with high quality shading,
    /// then output performance metrics to benchmark.log
    #[arg(long, short = 'b')]
    benchmark: bool,

    /// Output PNG to file instead of stdout
    #[arg(long, short = 'o', value_name = "OUTPUT")]
    output: Option<PathBuf>,

    /// Resolution for output (WxH format, e.g., 800x600). Default: terminal size for stdout, 800x600 for -o
    #[arg(long, short = 'r')]
    resolution: Option<String>,

    /// Representation mode (cartoon, ball-and-stick, surface, backbone)
    #[arg(long, default_value = "cartoon")]
    repr: String,

    /// Color scheme (chain, rainbow, secondary)
    #[arg(long, short = 'c', default_value = "chain")]
    color: String,

    /// Disable shading (flat colors)
    #[arg(long)]
    no_shading: bool,

    /// Background color (white, black, transparent)
    #[arg(long, default_value = "black")]
    bg: String,

    /// Force render backend (iterm2, half)
    #[arg(long)]
    backend: Option<String>,
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

    // Parse color scheme
    let color_scheme = match args.color.to_lowercase().as_str() {
        "chain" => render::ColorScheme::Chain,
        "rainbow" => render::ColorScheme::Rainbow,
        "secondary" | "ss" => render::ColorScheme::SecondaryStructure,
        _ => {
            eprintln!("Error: Unknown color scheme '{}'. Use: chain, rainbow, secondary", args.color);
            return ExitCode::from(1);
        }
    };

    // Parse background color
    let bg_color: Option<(u8, u8, u8)> = match args.bg.to_lowercase().as_str() {
        "white" => Some((255, 255, 255)),
        "black" => Some((0, 0, 0)),
        "transparent" | "none" => None,
        _ => {
            eprintln!("Error: Unknown background '{}'. Use: white, black, transparent", args.bg);
            return ExitCode::from(1);
        }
    };

    // Parse render backend
    let backend = match args.backend.as_deref() {
        Some(b) => match b.to_lowercase().as_str() {
            "iterm2" | "iterm" => Some(ui::RenderBackend::ITerm2),
            "half" | "half-block" | "halfblock" => Some(ui::RenderBackend::HalfBlock),
            _ => {
                eprintln!("Error: Unknown backend '{}'. Use: iterm2, half", b);
                return ExitCode::from(1);
            }
        },
        None => None, // Auto-detect
    };

    let options = ui::RenderOptions {
        representation: repr,
        color_scheme,
        shading: !args.no_shading,
        background: bg_color,
        backend,
    };

    // Run the appropriate mode
    let result = if let Some(output_path) = args.output {
        // Output to PNG file
        let (width, height) = match args.resolution.as_deref() {
            Some(res) => match parse_resolution(res) {
                Some(r) => r,
                None => {
                    eprintln!("Error: Invalid resolution format '{}'. Expected WxH (e.g., 800x600)", res);
                    return ExitCode::from(1);
                }
            },
            None => (800, 600), // Default for file output
        };
        ui::render_to_png(&args.file, molecule, &output_path, width, height, options)
    } else if args.benchmark {
        ui::run_benchmark(&args.file, molecule)
    } else if args.interactive {
        // Interactive mode with keyboard controls
        ui::run(&args.file, molecule)
    } else {
        // Default: render to stdout (auto-detect backend)
        let resolution = args.resolution.as_deref().and_then(parse_resolution);
        ui::render_to_stdout(&args.file, molecule, resolution, options)
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
