//! Atom representation and element data

use nalgebra::Vector3;

/// A single atom in a molecular structure
#[derive(Debug, Clone)]
pub struct Atom {
    /// Atom serial number
    pub serial: u32,
    /// Atom name (e.g., "CA", "N", "O")
    pub name: String,
    /// Alternate location indicator
    pub alt_loc: Option<char>,
    /// Residue name (e.g., "ALA", "GLY")
    pub residue_name: String,
    /// Chain identifier
    pub chain_id: char,
    /// Residue sequence number
    pub residue_seq: i32,
    /// Insertion code
    pub ins_code: Option<char>,
    /// 3D coordinates in Ångströms
    pub coord: Vector3<f32>,
    /// Occupancy (0.0-1.0)
    pub occupancy: f32,
    /// Temperature factor (B-factor)
    pub temp_factor: f32,
    /// Chemical element
    pub element: Element,
    /// Whether this is a heteroatom (HETATM vs ATOM)
    pub is_hetatm: bool,
}

impl Atom {
    /// Check if this is a water molecule
    pub fn is_water(&self) -> bool {
        matches!(self.residue_name.as_str(), "HOH" | "WAT" | "H2O" | "DOD")
    }

    /// Check if this is a backbone atom
    pub fn is_backbone(&self) -> bool {
        matches!(self.name.as_str(), "N" | "CA" | "C" | "O" | "P" | "O3'" | "O5'" | "C3'" | "C4'" | "C5'")
    }

    /// Get the Van der Waals radius for this atom
    pub fn vdw_radius(&self) -> f32 {
        self.element.vdw_radius()
    }
}

/// Chemical element with associated properties
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Element {
    H,
    C,
    N,
    O,
    S,
    P,
    Fe,
    Zn,
    Ca,
    Mg,
    Na,
    K,
    Cl,
    F,
    Br,
    I,
    Se,
    /// Unknown or unrecognized element
    Unknown,
}

impl Element {
    /// Parse element from string (1 or 2 character symbol)
    pub fn from_symbol(s: &str) -> Self {
        match s.trim().to_uppercase().as_str() {
            "H" => Element::H,
            "C" => Element::C,
            "N" => Element::N,
            "O" => Element::O,
            "S" => Element::S,
            "P" => Element::P,
            "FE" => Element::Fe,
            "ZN" => Element::Zn,
            "CA" => Element::Ca,
            "MG" => Element::Mg,
            "NA" => Element::Na,
            "K" => Element::K,
            "CL" => Element::Cl,
            "F" => Element::F,
            "BR" => Element::Br,
            "I" => Element::I,
            "SE" => Element::Se,
            _ => Element::Unknown,
        }
    }

    /// Get Van der Waals radius in Ångströms
    pub fn vdw_radius(&self) -> f32 {
        match self {
            Element::H => 1.20,
            Element::C => 1.70,
            Element::N => 1.55,
            Element::O => 1.52,
            Element::S => 1.80,
            Element::P => 1.80,
            Element::Fe => 1.40,
            Element::Zn => 1.39,
            Element::Ca => 1.97,
            Element::Mg => 1.73,
            Element::Na => 2.27,
            Element::K => 2.75,
            Element::Cl => 1.75,
            Element::F => 1.47,
            Element::Br => 1.85,
            Element::I => 1.98,
            Element::Se => 1.90,
            Element::Unknown => 1.50, // Default radius
        }
    }

    /// Get CPK color as RGB tuple
    pub fn cpk_color(&self) -> (u8, u8, u8) {
        match self {
            Element::H => (255, 255, 255),  // White
            Element::C => (144, 144, 144),  // Gray
            Element::N => (48, 80, 248),    // Blue
            Element::O => (255, 13, 13),    // Red
            Element::S => (255, 255, 48),   // Yellow
            Element::P => (255, 128, 0),    // Orange
            Element::Fe => (224, 102, 51),  // Orange-brown
            Element::Zn => (125, 128, 176), // Slate gray
            Element::Ca => (61, 255, 0),    // Green
            Element::Mg => (138, 255, 0),   // Light green
            Element::Na => (171, 92, 242),  // Purple
            Element::K => (143, 64, 212),   // Purple
            Element::Cl => (31, 240, 31),   // Green
            Element::F => (144, 224, 80),   // Light green
            Element::Br => (166, 41, 41),   // Dark red
            Element::I => (148, 0, 148),    // Purple
            Element::Se => (255, 161, 0),   // Orange
            Element::Unknown => (255, 20, 147), // Pink
        }
    }

    /// Get element symbol
    pub fn symbol(&self) -> &'static str {
        match self {
            Element::H => "H",
            Element::C => "C",
            Element::N => "N",
            Element::O => "O",
            Element::S => "S",
            Element::P => "P",
            Element::Fe => "Fe",
            Element::Zn => "Zn",
            Element::Ca => "Ca",
            Element::Mg => "Mg",
            Element::Na => "Na",
            Element::K => "K",
            Element::Cl => "Cl",
            Element::F => "F",
            Element::Br => "Br",
            Element::I => "I",
            Element::Se => "Se",
            Element::Unknown => "X",
        }
    }
}

impl Default for Element {
    fn default() -> Self {
        Element::Unknown
    }
}
