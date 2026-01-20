//! Rendering engine for molecular structures
//!
//! This module handles:
//! - Camera and projection transforms
//! - Braille character rasterization
//! - Different molecular representations

mod camera;
pub mod braille;
mod pixel;
pub mod surface;

pub use camera::{Camera, ProjectionMode};
pub use pixel::{PixelBuffer, downsample_2x, apply_edge_aa, apply_silhouette_edges, apply_ssao, apply_tone_mapping, fill_depth_gaps};
pub use surface::{generate_surface, Triangle, SurfaceAtom};

/// Representation style for rendering molecules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Representation {
    /// Backbone trace (Cα for proteins, P for nucleic acids)
    Backbone,
    /// Cartoon/ribbon (secondary structure visualization)
    #[default]
    Cartoon,
    /// Solvent excluded surface
    Surface,
}

impl Representation {
    /// Cycle to the next representation
    pub fn next(self) -> Self {
        match self {
            Representation::Backbone => Representation::Cartoon,
            Representation::Cartoon => Representation::Surface,
            Representation::Surface => Representation::Backbone,
        }
    }

    /// Get the name of this representation
    pub fn name(&self) -> &'static str {
        match self {
            Representation::Backbone => "Backbone",
            Representation::Cartoon => "Cartoon",
            Representation::Surface => "Surface",
        }
    }
}

/// Color scheme for rendering molecules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorScheme {
    /// Distinct color per chain
    #[default]
    Chain,
    /// N-terminus (blue) to C-terminus (red) gradient
    Rainbow,
    /// Helix=magenta, Sheet=yellow, Coil=white
    SecondaryStructure,
}

impl ColorScheme {
    /// Cycle to the next color scheme
    pub fn next(self) -> Self {
        match self {
            ColorScheme::Chain => ColorScheme::Rainbow,
            ColorScheme::Rainbow => ColorScheme::SecondaryStructure,
            ColorScheme::SecondaryStructure => ColorScheme::Chain,
        }
    }

    /// Get the name of this color scheme
    pub fn name(&self) -> &'static str {
        match self {
            ColorScheme::Chain => "Chain",
            ColorScheme::Rainbow => "Rainbow",
            ColorScheme::SecondaryStructure => "Secondary Structure",
        }
    }
}

/// Get a color for a chain identifier
pub fn chain_color(chain_id: char) -> (u8, u8, u8) {
    // PyMOL/ChimeraX-style pleasant palette
    const CHAIN_COLORS: [(u8, u8, u8); 10] = [
        (30, 144, 255),  // A - Dodger blue
        (255, 165, 0),   // B - Orange
        (50, 205, 50),   // C - Lime green
        (255, 99, 71),   // D - Tomato red
        (138, 43, 226),  // E - Blue violet
        (0, 206, 209),   // F - Dark turquoise
        (255, 215, 0),   // G - Gold
        (199, 21, 133),  // H - Medium violet red
        (0, 191, 255),   // I - Deep sky blue
        (255, 105, 180), // J - Hot pink
    ];

    let idx = match chain_id {
        'A'..='J' => (chain_id as u8 - b'A') as usize,
        'a'..='j' => (chain_id as u8 - b'a') as usize,
        _ => 0,
    };

    CHAIN_COLORS[idx % CHAIN_COLORS.len()]
}

/// Get a rainbow color based on position (0.0 to 1.0)
pub fn rainbow_color(t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);

    // Blue (0.0) -> Cyan -> Green -> Yellow -> Red (1.0)
    let r: u8;
    let g: u8;
    let b: u8;

    if t < 0.25 {
        let f = t * 4.0;
        r = 0;
        g = (255.0 * f) as u8;
        b = 255;
    } else if t < 0.5 {
        let f = (t - 0.25) * 4.0;
        r = 0;
        g = 255;
        b = (255.0 * (1.0 - f)) as u8;
    } else if t < 0.75 {
        let f = (t - 0.5) * 4.0;
        r = (255.0 * f) as u8;
        g = 255;
        b = 0;
    } else {
        let f = (t - 0.75) * 4.0;
        r = 255;
        g = (255.0 * (1.0 - f)) as u8;
        b = 0;
    }

    (r, g, b)
}
