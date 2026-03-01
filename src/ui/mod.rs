//! Terminal UI and application loop
//!
//! Handles keyboard/mouse input, rendering to terminal, and HUD display.

mod output;

pub use output::RenderBackend;

use crate::molecule::{Assembly, Molecule, SecondaryStructure};
use crate::render::{PixelBuffer, Camera, ColorScheme, Representation, chain_color, rainbow_color, apply_edge_aa, apply_silhouette_edges, apply_ssao, apply_tone_mapping, fill_depth_gaps, generate_surface, SurfaceAtom};
use rayon::prelude::*;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyModifiers, MouseButton, MouseEvent, MouseEventKind},
    execute,
    terminal,
};
use ratatui::{
    backend::CrosstermBackend,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Frame, Terminal,
};
use std::collections::HashMap;
use std::io::{self, Stdout};
use std::path::Path;
use std::time::{Duration, Instant};
use thiserror::Error;
use nalgebra::{Vector2, Vector3};
use output::detect_backend;

/// Options for PNG rendering
pub struct RenderOptions {
    pub representation: Representation,
    pub color_scheme: ColorScheme,
    pub shading: bool,
    pub background: Option<(u8, u8, u8)>,
    pub backend: Option<RenderBackend>,
}

/// UI-related errors
#[derive(Error, Debug)]
pub enum UiError {
    #[error("Terminal error: {0}")]
    TerminalError(#[from] io::Error),

    #[error("Crossterm error: {0}")]
    CrosstermError(String),
}

/// Alternate location display mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AltLocMode {
    A,
    B,
    All,
}

impl AltLocMode {
    fn next(self) -> Self {
        match self {
            AltLocMode::A => AltLocMode::B,
            AltLocMode::B => AltLocMode::All,
            AltLocMode::All => AltLocMode::A,
        }
    }

    fn label(self) -> &'static str {
        match self {
            AltLocMode::A => "A",
            AltLocMode::B => "B",
            AltLocMode::All => "All",
        }
    }
}

const TILE_SIZE: usize = 64;
const TILE_THRESHOLD: usize = 128;

#[derive(Clone, Copy)]
struct ProjInfo {
    x: f32,
    y: f32,
    z: f32,
    size_scale: f32,
    color: (u8, u8, u8),
}

#[derive(Clone, Copy)]
struct Bounds {
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
}

impl Bounds {
    fn from_circle(cx: f32, cy: f32, radius: f32) -> Self {
        let min_x = (cx - radius).floor() as i32;
        let max_x = (cx + radius).ceil() as i32;
        let min_y = (cy - radius).floor() as i32;
        let max_y = (cy + radius).ceil() as i32;
        Self { min_x, max_x, min_y, max_y }
    }

    fn from_segment(x0: f32, y0: f32, x1: f32, y1: f32, radius: f32) -> Self {
        let min_x = (x0.min(x1) - radius).floor() as i32;
        let max_x = (x0.max(x1) + radius).ceil() as i32;
        let min_y = (y0.min(y1) - radius).floor() as i32;
        let max_y = (y0.max(y1) + radius).ceil() as i32;
        Self { min_x, max_x, min_y, max_y }
    }

    fn intersects(&self, other: &Bounds) -> bool {
        !(self.max_x < other.min_x
            || self.min_x > other.max_x
            || self.max_y < other.min_y
            || self.min_y > other.max_y)
    }
}

#[derive(Clone, Copy)]
struct Tile {
    x0: i32,
    y0: i32,
    width: usize,
    height: usize,
}

impl Tile {
    fn bounds(&self) -> Bounds {
        let max_x = self.x0 + self.width as i32 - 1;
        let max_y = self.y0 + self.height as i32 - 1;
        Bounds {
            min_x: self.x0,
            max_x,
            min_y: self.y0,
            max_y,
        }
    }
}

enum Primitive {
    Sphere { x: f32, y: f32, z: f32, radius: f32, color: (u8, u8, u8), bounds: Bounds },
    Cylinder { x0: f32, y0: f32, z0: f32, x1: f32, y1: f32, z1: f32, radius: f32, color: (u8, u8, u8), bounds: Bounds },
    Line { x0: f32, y0: f32, z0: f32, x1: f32, y1: f32, z1: f32, color: (u8, u8, u8), bounds: Bounds },
    Circle { x: f32, y: f32, z: f32, radius: f32, color: (u8, u8, u8), bounds: Bounds },
}

impl Primitive {
    fn sphere(x: f32, y: f32, z: f32, radius: f32, color: (u8, u8, u8)) -> Self {
        let bounds = Bounds::from_circle(x, y, radius);
        Primitive::Sphere { x, y, z, radius, color, bounds }
    }

    fn cylinder(x0: f32, y0: f32, z0: f32, x1: f32, y1: f32, z1: f32, radius: f32, color: (u8, u8, u8)) -> Self {
        let bounds = Bounds::from_segment(x0, y0, x1, y1, radius);
        Primitive::Cylinder { x0, y0, z0, x1, y1, z1, radius, color, bounds }
    }

    fn line(x0: f32, y0: f32, z0: f32, x1: f32, y1: f32, z1: f32, color: (u8, u8, u8)) -> Self {
        let bounds = Bounds::from_segment(x0, y0, x1, y1, 0.5);
        Primitive::Line { x0, y0, z0, x1, y1, z1, color, bounds }
    }

    fn circle(x: f32, y: f32, z: f32, radius: f32, color: (u8, u8, u8)) -> Self {
        let bounds = Bounds::from_circle(x, y, radius);
        Primitive::Circle { x, y, z, radius, color, bounds }
    }

    fn bounds(&self) -> Bounds {
        match self {
            Primitive::Sphere { bounds, .. } => *bounds,
            Primitive::Cylinder { bounds, .. } => *bounds,
            Primitive::Line { bounds, .. } => *bounds,
            Primitive::Circle { bounds, .. } => *bounds,
        }
    }

    fn draw_offset(&self, buffer: &mut PixelBuffer, offset_x: i32, offset_y: i32) {
        let ox = offset_x as f32;
        let oy = offset_y as f32;
        match *self {
            Primitive::Sphere { x, y, z, radius, color, .. } => {
                buffer.draw_sphere_shaded(x - ox, y - oy, z, radius, color);
            }
            Primitive::Cylinder { x0, y0, z0, x1, y1, z1, radius, color, .. } => {
                buffer.draw_cylinder_shaded(x0 - ox, y0 - oy, z0, x1 - ox, y1 - oy, z1, radius, color);
            }
            Primitive::Line { x0, y0, z0, x1, y1, z1, color, .. } => {
                buffer.draw_line(x0 - ox, y0 - oy, z0, x1 - ox, y1 - oy, z1, color);
            }
            Primitive::Circle { x, y, z, radius, color, .. } => {
                buffer.draw_circle(x - ox, y - oy, z, radius, color);
            }
        }
    }
}

fn render_primitives_tiled(buffer: &mut PixelBuffer, primitives: &[Primitive]) {
    if primitives.is_empty() {
        return;
    }

    let width = buffer.width();
    let height = buffer.height();
    let tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    let tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;

    let mut tiles = Vec::with_capacity(tiles_x * tiles_y);
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x0 = (tx * TILE_SIZE) as i32;
            let y0 = (ty * TILE_SIZE) as i32;
            let tile_w = TILE_SIZE.min(width.saturating_sub(tx * TILE_SIZE));
            let tile_h = TILE_SIZE.min(height.saturating_sub(ty * TILE_SIZE));
            if tile_w == 0 || tile_h == 0 {
                continue;
            }
            tiles.push(Tile { x0, y0, width: tile_w, height: tile_h });
        }
    }

    let rendered: Vec<(usize, usize, PixelBuffer)> = tiles
        .par_iter()
        .map(|tile| {
            let mut local = PixelBuffer::new(tile.width, tile.height);
            let tile_bounds = tile.bounds();

            for prim in primitives {
                if prim.bounds().intersects(&tile_bounds) {
                    prim.draw_offset(&mut local, tile.x0, tile.y0);
                }
            }

            (tile.x0 as usize, tile.y0 as usize, local)
        })
        .collect();

    for (x0, y0, local) in rendered {
        buffer.blit_from(&local, x0, y0);
    }
}

/// Application state
struct App {
    /// The molecule being viewed
    molecule: Molecule,
    /// File name for display
    filename: String,
    /// Camera state
    camera: Camera,
    /// Current representation
    representation: Representation,
    /// Current color scheme
    color_scheme: ColorScheme,
    /// Whether HUD is visible
    show_hud: bool,
    /// Whether help overlay is visible
    show_help: bool,
    /// Whether ligands are visible
    show_ligands: bool,
    /// Whether waters are visible
    show_waters: bool,
    /// Whether shading is enabled
    shading_enabled: bool,
    /// Whether auto-spin is enabled
    auto_spin: bool,
    /// Alternate location display mode
    alt_loc_mode: AltLocMode,
    /// Default assembly when file does not define any
    default_assembly: Assembly,
    /// Current assembly index
    assembly_index: usize,
    /// Chain visibility map (supports any chain ID character)
    chain_visible: HashMap<char, bool>,
    /// Mouse drag state
    mouse_drag: Option<(u16, u16)>,
    /// Whether the view needs to be redrawn
    needs_redraw: bool,
    /// Current render backend
    backend: RenderBackend,
    /// Whether backend changed (requires screen clear)
    backend_changed: bool,
    /// Whether to quit
    should_quit: bool,
    /// Last time an image was sent to terminal (for frame rate limiting)
    last_image_sent: Instant,
    /// Minimum interval between image outputs (caps terminal I/O)
    image_interval: Duration,
    /// Last time camera/view was changed (for adaptive resolution)
    last_interaction: Instant,
}

/// Sanitize a string for safe display in terminal
/// Removes control characters that could be used for terminal escape injection
fn sanitize_for_display(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}

impl App {
    fn new(path: &Path, molecule: Molecule, backend: RenderBackend) -> Self {
        let filename = sanitize_for_display(
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown"),
        );

        let center = molecule.center();
        let camera = Camera::new(center);

        // Initialize chain visibility map with all chains visible
        let chain_visible: HashMap<char, bool> = molecule
            .chains
            .iter()
            .map(|&c| (c, true))
            .collect();

        let mut app = Self {
            molecule,
            filename,
            camera,
            representation: Representation::default(),
            color_scheme: ColorScheme::default(),
            show_hud: true,
            show_help: false,
            show_ligands: false,
            show_waters: false,
            shading_enabled: true,
            auto_spin: false,
            alt_loc_mode: AltLocMode::A,
            default_assembly: Assembly::single_identity(),
            assembly_index: 0,
            chain_visible,
            mouse_drag: None,
            needs_redraw: true,
            backend,
            backend_changed: false,
            should_quit: false,
            last_image_sent: Instant::now(),
            // Protocol-specific image intervals to avoid overwhelming terminal
            image_interval: match backend {
                RenderBackend::ITerm2 => Duration::from_millis(50), // ~20 FPS
                RenderBackend::HalfBlock => Duration::from_millis(16), // ~60 FPS
            },
            last_interaction: Instant::now(),
        };

        app.fit_camera_to_current_assembly();
        app
    }

    fn handle_key(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        // Track keyboard interaction for adaptive resolution and redraw
        // (skip for quit which doesn't affect view)
        if !matches!(code, KeyCode::Char('q') | KeyCode::Esc) {
            self.last_interaction = Instant::now();
            self.needs_redraw = true;
        }
        match code {
            KeyCode::Char('q') | KeyCode::Esc => {
                self.should_quit = true;
            }
            KeyCode::Tab => {
                self.representation = self.representation.next();
            }
            KeyCode::Char('c') => {
                self.color_scheme = self.color_scheme.next();
            }
            KeyCode::Char('s') => {
                self.shading_enabled = !self.shading_enabled;
            }
            KeyCode::Char('p') => {
                self.auto_spin = !self.auto_spin;
            }
            KeyCode::Char('/') => {
                let count = self.molecule.assembly_count();
                if count > 1 {
                    self.assembly_index = (self.assembly_index + 1) % count;
                    let (min, max) = self.current_assembly_bounds();
                    let center = (min + max) / 2.0;
                    self.camera.center = center;
                }
            }
            KeyCode::Char('\'') => {
                self.alt_loc_mode = self.alt_loc_mode.next();
            }
            KeyCode::Char('l') => {
                self.show_ligands = !self.show_ligands;
            }
            KeyCode::Char('h') | KeyCode::Char('H') => {
                self.show_help = !self.show_help;
            }
            KeyCode::F(1) => {
                self.show_hud = !self.show_hud;
            }
            KeyCode::F(3) => {
                self.show_waters = !self.show_waters;
            }
            KeyCode::Char('v') => {
                self.camera.toggle_projection();
            }
            KeyCode::Char('b') => {
                self.cycle_backend();
            }
            KeyCode::Char('0') => {
                self.fit_camera_to_current_assembly();
            }
            KeyCode::Char('[') => {
                self.camera.zoom_by(0.9); // Zoom out
            }
            KeyCode::Char(']') => {
                self.camera.zoom_by(1.1); // Zoom in
            }
            KeyCode::Up => {
                if modifiers.contains(KeyModifiers::SHIFT) {
                    // Rotate up around X axis
                    self.camera.trackball_rotate(
                        Vector2::new(0.0, 0.0),
                        Vector2::new(0.0, -0.02),
                    );
                } else {
                    self.camera.pan(Vector2::new(0.0, -0.1));
                }
            }
            KeyCode::Down => {
                if modifiers.contains(KeyModifiers::SHIFT) {
                    // Rotate down around X axis
                    self.camera.trackball_rotate(
                        Vector2::new(0.0, 0.0),
                        Vector2::new(0.0, 0.02),
                    );
                } else {
                    self.camera.pan(Vector2::new(0.0, 0.1));
                }
            }
            KeyCode::Left => {
                if modifiers.contains(KeyModifiers::SHIFT) {
                    // Rotate left around Y axis
                    self.camera.trackball_rotate(
                        Vector2::new(0.0, 0.0),
                        Vector2::new(-0.02, 0.0),
                    );
                } else {
                    self.camera.pan(Vector2::new(-0.1, 0.0));
                }
            }
            KeyCode::Right => {
                if modifiers.contains(KeyModifiers::SHIFT) {
                    // Rotate right around Y axis
                    self.camera.trackball_rotate(
                        Vector2::new(0.0, 0.0),
                        Vector2::new(0.02, 0.0),
                    );
                } else {
                    self.camera.pan(Vector2::new(0.1, 0.0));
                }
            }
            KeyCode::Char(c) if c.is_ascii_uppercase() => {
                // Toggle chain visibility if this chain exists
                if let Some(visible) = self.chain_visible.get_mut(&c) {
                    *visible = !*visible;
                }
            }
            _ => {}
        }
    }

    fn handle_mouse(&mut self, event: MouseEvent, width: u16, height: u16) {
        match event.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                self.mouse_drag = Some((event.column, event.row));
                self.last_interaction = Instant::now();
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                if let Some((prev_x, prev_y)) = self.mouse_drag {
                    // Convert to normalized coordinates (-1 to 1) with sensitivity scaling
                    const ROTATION_SENSITIVITY: f32 = 0.4; // Slower rotation
                    let prev = Vector2::new(
                        (prev_x as f32 / width as f32) * 2.0 - 1.0,
                        (prev_y as f32 / height as f32) * 2.0 - 1.0,
                    );
                    let curr_raw = Vector2::new(
                        (event.column as f32 / width as f32) * 2.0 - 1.0,
                        (event.row as f32 / height as f32) * 2.0 - 1.0,
                    );
                    // Scale the delta for slower rotation
                    let delta = (curr_raw - prev) * ROTATION_SENSITIVITY;
                    let curr = prev + delta;
                    self.camera.trackball_rotate(prev, curr);
                    self.mouse_drag = Some((event.column, event.row));
                    self.last_interaction = Instant::now();
                    self.needs_redraw = true;
                }
            }
            MouseEventKind::Up(MouseButton::Left) => {
                self.mouse_drag = None;
            }
            MouseEventKind::ScrollUp => {
                self.camera.zoom_by(1.1);
                self.last_interaction = Instant::now();
                self.needs_redraw = true;
            }
            MouseEventKind::ScrollDown => {
                self.camera.zoom_by(0.9);
                self.last_interaction = Instant::now();
                self.needs_redraw = true;
            }
            _ => {}
        }
    }

    fn is_atom_visible(&self, atom: &crate::molecule::Atom) -> bool {
        // Check chain visibility using the HashMap
        if let Some(&visible) = self.chain_visible.get(&atom.chain_id) {
            if !visible {
                return false;
            }
        }

        // Check alternate location visibility
        if let Some(alt) = atom.alt_loc {
            let alt = alt.to_ascii_uppercase();
            match self.alt_loc_mode {
                AltLocMode::A => {
                    if alt != 'A' {
                        return false;
                    }
                }
                AltLocMode::B => {
                    if alt != 'B' {
                        return false;
                    }
                }
                AltLocMode::All => {}
            }
        }

        // Check heteroatom visibility
        if atom.is_hetatm {
            if atom.is_water() {
                return self.show_waters;
            } else {
                return self.show_ligands;
            }
        }

        true
    }

    /// Get the secondary structure type for a residue
    fn get_secondary_structure(&self, chain_id: char, residue_seq: i32) -> SecondaryStructure {
        for ss in &self.molecule.secondary_structure {
            if ss.contains(chain_id, residue_seq) {
                return ss.ss_type;
            }
        }
        SecondaryStructure::Coil
    }

    fn render_molecule(&self, buffer: &mut PixelBuffer, _pixels_per_cell: (usize, usize)) {
        let pixel_width = buffer.width() as f32;
        let pixel_height = buffer.height() as f32;
        let center_x = pixel_width / 2.0;
        let center_y = pixel_height / 2.0;

        // Use uniform scaling based on pixel dimensions to maintain aspect ratio
        // Scale factor of 0.45 means molecule spans ~90% of the smaller dimension
        // (±0.45 of half = 45% per side = 90% total)
        let base_scale = pixel_width.min(pixel_height) * 0.45;
        let scale_x = base_scale;
        let scale_y = base_scale;
        let scale = base_scale;
        // Screen scale for perspective projection - based on viewport size only
        // Zoom is handled separately in camera.project_with_scale via self.zoom
        let screen_scale = pixel_width.min(pixel_height) * 0.5;

        // Collect visible atoms with their screen positions per assembly instance
        // Now includes size_scale for perspective-correct sizing
        let mut projected_instances: Vec<Vec<(usize, ProjInfo)>> = Vec::new();
        let mut z_min = f32::INFINITY;
        let mut z_max = f32::NEG_INFINITY;

        for instance in &self.current_assembly().instances {
            let mut projected: Vec<(usize, ProjInfo)> = Vec::with_capacity(self.molecule.atoms.len());
            for (idx, atom) in self.molecule.atoms.iter().enumerate() {
                if !self.is_atom_visible(atom) {
                    continue;
                }
                if !instance.applies_to_chain(atom.chain_id) {
                    continue;
                }

                let world = instance.transform.apply(atom.coord);
                let (screen_pos, z, size_scale) = self.camera.project_with_scale(world, screen_scale);
                let sx = center_x + screen_pos.x * scale_x;
                let sy = center_y + screen_pos.y * scale_y;

                z_min = z_min.min(z);
                z_max = z_max.max(z);

                // Determine color based on scheme
                let color = self.get_atom_color(atom, idx);

                projected.push((idx, ProjInfo { x: sx, y: sy, z, size_scale, color }));
            }
            projected_instances.push(projected);
        }

        for projected in &projected_instances {
            // Render based on representation
            match self.representation {
                Representation::Backbone => {
                    self.render_backbone(buffer, projected, scale, z_min, z_max);
                }
                Representation::Cartoon => {
                    self.render_cartoon(buffer, projected, scale, z_min, z_max);
                }
                Representation::Surface => {
                    self.render_surface(buffer, projected, scale, z_min, z_max);
                }
            }
        }
    }

    fn get_atom_color(&self, atom: &crate::molecule::Atom, atom_idx: usize) -> (u8, u8, u8) {
        match self.color_scheme {
            ColorScheme::Chain => chain_color(atom.chain_id),
            ColorScheme::Rainbow => {
                let t = atom_idx as f32 / self.molecule.atoms.len().max(1) as f32;
                rainbow_color(t)
            }
            ColorScheme::SecondaryStructure => {
                // Check secondary structure assignment
                // Colors match PyMOL/ChimeraX conventions
                for ss in &self.molecule.secondary_structure {
                    if ss.contains(atom.chain_id, atom.residue_seq) {
                        return match ss.ss_type {
                            SecondaryStructure::Helix(_) => (0, 191, 255), // Cyan/light blue
                            SecondaryStructure::Sheet => (255, 200, 50),   // Golden yellow
                            SecondaryStructure::Coil => (200, 200, 200),   // Light gray
                        };
                    }
                }
                (200, 200, 200) // Default light gray for coil
            }
        }
    }

    fn render_backbone(
        &self,
        buffer: &mut PixelBuffer,
        projected: &[(usize, ProjInfo)],
        _scale: f32,
        z_min: f32,
        z_max: f32,
    ) {
        use crate::render::braille::depth_cue;

        // Get backbone atoms (CA for proteins, P for nucleic acids)
        let backbone_indices: Vec<usize> = self
            .molecule
            .atoms
            .iter()
            .enumerate()
            .filter(|(_, a)| (a.name == "CA" || a.name == "P") && self.is_atom_visible(a))
            .map(|(i, _)| i)
            .collect();

        // Create a map from atom index to projected position (includes size_scale)
        let mut proj_map = vec![None; self.molecule.atoms.len()];
        for (idx, info) in projected {
            proj_map[*idx] = Some(*info);
        }

        // Draw lines between consecutive backbone atoms in same chain
        let mut prev: Option<(char, i32, f32, f32, f32, (u8, u8, u8))> = None;

        for &idx in &backbone_indices {
            let atom = &self.molecule.atoms[idx];
            if let Some(info) = proj_map[idx] {
                let color = if self.shading_enabled {
                    depth_cue(info.color, info.z, z_max, z_min)
                } else {
                    info.color
                };

                if let Some((prev_chain, prev_seq, px, py, pz, _pcolor)) = prev {
                    // Only connect if same chain and consecutive residue
                    if atom.chain_id == prev_chain && atom.residue_seq == prev_seq + 1 {
                        buffer.draw_line(px, py, pz, info.x, info.y, info.z, color);
                    }
                }

                prev = Some((atom.chain_id, atom.residue_seq, info.x, info.y, info.z, color));
            }
        }
    }

    fn render_surface(
        &self,
        buffer: &mut PixelBuffer,
        projected: &[(usize, ProjInfo)],
        scale: f32,
        _z_min: f32,
        _z_max: f32,
    ) {
        use nalgebra::Vector3;

        // Build atom list for surface generation
        // Map atom index to color from projected list
        let mut color_map: std::collections::HashMap<usize, (u8, u8, u8)> = std::collections::HashMap::new();
        for (idx, info) in projected {
            color_map.insert(*idx, info.color);
        }

        let surface_atoms: Vec<SurfaceAtom> = self.molecule.atoms
            .iter()
            .enumerate()
            .filter(|(idx, _)| color_map.contains_key(idx))
            .map(|(idx, atom)| SurfaceAtom {
                pos: atom.coord,
                radius: atom.vdw_radius(),
                color: color_map.get(&idx).copied().unwrap_or((128, 128, 128)),
                chain_id: atom.chain_id,
            })
            .collect();

        if surface_atoms.is_empty() {
            return;
        }

        // Generate surface mesh using marching cubes
        // probe_radius: 1.4 Å (water molecule)
        // grid_spacing: 0.8 Å for good quality (lower = higher quality but slower)
        let probe_radius = 1.4;
        let grid_spacing = 0.8;
        let triangles = generate_surface(&surface_atoms, probe_radius, grid_spacing);

        if triangles.is_empty() {
            return;
        }

        // Find dominant color per region (use first atom's chain color as base)
        // For proper coloring, we'd need to assign colors per-vertex based on nearest atom
        let base_color = surface_atoms[0].color;

        // Get camera transform matrices
        let width = buffer.width() as f32;
        let height = buffer.height() as f32;
        let center_x = width / 2.0;
        let center_y = height / 2.0;

        // Project and render each triangle
        for tri in &triangles {
            // Transform vertices from world to camera space
            let v0_cam = self.camera.transform_point(&tri.v0);
            let v1_cam = self.camera.transform_point(&tri.v1);
            let v2_cam = self.camera.transform_point(&tri.v2);

            // Transform normals (rotate only, no translation)
            let n0_cam = self.camera.transform_normal(&tri.n0);
            let n1_cam = self.camera.transform_normal(&tri.n1);
            let n2_cam = self.camera.transform_normal(&tri.n2);

            // Project to screen space
            let (x0, y0, z0) = self.camera.project_point(&v0_cam, center_x, center_y, scale);
            let (x1, y1, z1) = self.camera.project_point(&v1_cam, center_x, center_y, scale);
            let (x2, y2, z2) = self.camera.project_point(&v2_cam, center_x, center_y, scale);

            // Backface culling - skip triangles facing away from camera
            // Compute face normal in screen space
            let e1x = x1 - x0;
            let e1y = y1 - y0;
            let e2x = x2 - x0;
            let e2y = y2 - y0;
            let cross_z = e1x * e2y - e1y * e2x;
            if cross_z < 0.0 {
                continue; // Back-facing
            }

            // Find nearest atom to triangle center for coloring
            let tri_center = (tri.v0 + tri.v1 + tri.v2) / 3.0;
            let mut best_color = base_color;
            let mut best_dist = f32::MAX;
            for atom in &surface_atoms {
                let dist = (atom.pos - tri_center).norm();
                if dist < best_dist {
                    best_dist = dist;
                    best_color = atom.color;
                }
            }

            // Draw the triangle with smooth shading
            buffer.draw_triangle_shaded(
                x0, y0, z0,
                x1, y1, z1,
                x2, y2, z2,
                n0_cam.x, n0_cam.y, n0_cam.z,
                n1_cam.x, n1_cam.y, n1_cam.z,
                n2_cam.x, n2_cam.y, n2_cam.z,
                best_color,
            );
        }
    }

    /// Render cartoon representation with secondary structure elements
    fn render_cartoon(
        &self,
        buffer: &mut PixelBuffer,
        projected: &[(usize, ProjInfo)],
        _scale: f32,
        z_min: f32,
        z_max: f32,
    ) {
        // Get backbone atoms (CA for proteins, P for nucleic acids)
        let backbone_indices: Vec<usize> = self
            .molecule
            .atoms
            .iter()
            .enumerate()
            .filter(|(_, a)| (a.name == "CA" || a.name == "P") && self.is_atom_visible(a))
            .map(|(i, _)| i)
            .collect();

        // Create a map from atom index to projected position (includes size_scale)
        let mut proj_map = vec![None; self.molecule.atoms.len()];
        for (idx, info) in projected {
            proj_map[*idx] = Some(*info);
        }

        // Size scale for ribbon widths - based on screen size with gentle molecule scaling
        let pixel_width = buffer.width() as f32;
        let pixel_height = buffer.height() as f32;
        let screen_scale = pixel_height.min(pixel_width) / 50.0;

        // Gentle scale adjustment for very large molecules only
        let atom_count = self.molecule.atoms.len() as f32;
        let complexity_scale = if atom_count > 2000.0 {
            (2000.0 / atom_count).sqrt().max(0.5)
        } else {
            1.0
        };

        let ribbon_scale = screen_scale * complexity_scale;
        let helix_width = (1.8 * ribbon_scale).max(4.0).min(20.0);
        let sheet_width = (2.2 * ribbon_scale).max(5.0).min(25.0);
        let coil_width = (0.6 * ribbon_scale).max(1.5).min(10.0);
        let arrow_length = (sheet_width * 0.6).max(4.0).min(12.0);
        let arrow_width = (sheet_width * 0.8).max(5.0).min(15.0);

        // For spline rendering, we need to collect chain segments
        if self.shading_enabled {
            // Spline-based rendering for hi-fi mode
            self.render_cartoon_spline(
                buffer, &backbone_indices, &proj_map,
                helix_width, sheet_width, coil_width,
                arrow_length, arrow_width, z_min, z_max,
            );
        } else {
            // Original linear rendering
            self.render_cartoon_linear(
                buffer, &backbone_indices, &proj_map,
                helix_width, sheet_width, coil_width,
                arrow_length, arrow_width, z_min, z_max,
            );
        }
    }

    /// Linear cartoon rendering (original method)
    fn render_cartoon_linear(
        &self,
        buffer: &mut PixelBuffer,
        backbone_indices: &[usize],
        proj_map: &[Option<ProjInfo>],
        helix_width: f32,
        sheet_width: f32,
        coil_width: f32,
        arrow_length: f32,
        arrow_width: f32,
        z_min: f32,
        z_max: f32,
    ) {
        use crate::render::braille::depth_cue;

        let mut prev: Option<(char, i32, f32, f32, f32, (u8, u8, u8), SecondaryStructure)> = None;
        let mut last_segment_dir: Option<Vector2<f32>> = None;

        for &idx in backbone_indices {
            let atom = &self.molecule.atoms[idx];
            if let Some(info) = proj_map[idx] {
                let ss = self.get_secondary_structure(atom.chain_id, atom.residue_seq);
                let color = depth_cue(info.color, info.z, z_max, z_min);

                if let Some((prev_chain, prev_seq, px, py, pz, prev_color, prev_ss)) = prev {
                    if atom.chain_id == prev_chain && atom.residue_seq == prev_seq + 1 {
                        let segment_dir = Vector2::new(info.x - px, info.y - py);
                        last_segment_dir = Some(segment_dir);

                        // Use cylinder rendering for all - handles junctions properly
                        let width = match (&prev_ss, &ss) {
                            (SecondaryStructure::Sheet, _) | (_, SecondaryStructure::Sheet) => sheet_width,
                            (SecondaryStructure::Helix(_), _) | (_, SecondaryStructure::Helix(_)) => helix_width,
                            _ => coil_width,
                        };
                        self.draw_ribbon(buffer, px, py, pz, info.x, info.y, info.z, color, width);

                        if matches!(prev_ss, SecondaryStructure::Sheet) && !matches!(ss, SecondaryStructure::Sheet) {
                            buffer.draw_sheet_arrow(px, py, pz, segment_dir.x, segment_dir.y, arrow_length, arrow_width, prev_color);
                        }
                    } else {
                        last_segment_dir = None;
                    }
                }
                prev = Some((atom.chain_id, atom.residue_seq, info.x, info.y, info.z, color, ss));
            }
        }

        if let (Some((_, _, px, py, pz, color, ss)), Some(dir)) = (prev, last_segment_dir) {
            if matches!(ss, SecondaryStructure::Sheet) {
                buffer.draw_sheet_arrow(px, py, pz, dir.x, dir.y, arrow_length, arrow_width, color);
            }
        }
    }

    /// Spline-based cartoon rendering for hi-fi mode
    fn render_cartoon_spline(
        &self,
        buffer: &mut PixelBuffer,
        backbone_indices: &[usize],
        proj_map: &[Option<ProjInfo>],
        helix_width: f32,
        sheet_width: f32,
        coil_width: f32,
        arrow_length: f32,
        arrow_width: f32,
        z_min: f32,
        z_max: f32,
    ) {
        use crate::render::braille::depth_cue;

        // Collect chain segments with positions
        let mut segments: Vec<(f32, f32, f32, (u8, u8, u8), SecondaryStructure, char, i32)> = Vec::new();

        for &idx in backbone_indices {
            let atom = &self.molecule.atoms[idx];
            if let Some(info) = proj_map[idx] {
                let ss = self.get_secondary_structure(atom.chain_id, atom.residue_seq);
                let color = depth_cue(info.color, info.z, z_max, z_min);
                segments.push((info.x, info.y, info.z, color, ss, atom.chain_id, atom.residue_seq));
            }
        }

        if segments.len() < 2 {
            return;
        }

        // Draw spline through consecutive segments
        for i in 0..segments.len() - 1 {
            let (x1, y1, z1, c1, ss1, chain1, seq1) = segments[i];
            let (x2, y2, z2, _c2, ss2, chain2, seq2) = segments[i + 1];

            // Only connect consecutive residues in same chain
            if chain1 != chain2 || seq2 != seq1 + 1 {
                continue;
            }

            // Get control points for Catmull-Rom spline
            let p0 = if i > 0 && segments[i - 1].5 == chain1 && segments[i - 1].6 == seq1 - 1 {
                (segments[i - 1].0, segments[i - 1].1, segments[i - 1].2)
            } else {
                // Reflect p1 around p0 direction
                (2.0 * x1 - x2, 2.0 * y1 - y2, 2.0 * z1 - z2)
            };

            let p3 = if i + 2 < segments.len() && segments[i + 2].5 == chain2 && segments[i + 2].6 == seq2 + 1 {
                (segments[i + 2].0, segments[i + 2].1, segments[i + 2].2)
            } else {
                // Reflect p2 around p3 direction
                (2.0 * x2 - x1, 2.0 * y2 - y1, 2.0 * z2 - z1)
            };

            let p1 = (x1, y1, z1);
            let p2 = (x2, y2, z2);

            // Determine width based on secondary structure
            // Use cylinder rendering for all - handles junctions properly
            let width = match (&ss1, &ss2) {
                (SecondaryStructure::Sheet, _) | (_, SecondaryStructure::Sheet) => sheet_width,
                (SecondaryStructure::Helix(_), _) | (_, SecondaryStructure::Helix(_)) => helix_width,
                _ => coil_width,
            };
            self.draw_spline_ribbon_shaded(buffer, p0, p1, p2, p3, c1, width);

            // Draw 3D arrowhead at end of sheets
            if matches!(ss1, SecondaryStructure::Sheet) && !matches!(ss2, SecondaryStructure::Sheet) {
                let dir_x = x2 - x1;
                let dir_y = y2 - y1;
                buffer.draw_sheet_arrow(x1, y1, z1, dir_x, dir_y, arrow_length, arrow_width, c1);
            }
        }

        // Handle final arrowhead
        if segments.len() >= 2 {
            let last = &segments[segments.len() - 1];
            let prev = &segments[segments.len() - 2];
            if last.5 == prev.5 && last.6 == prev.6 + 1 && matches!(last.4, SecondaryStructure::Sheet) {
                let dir_x = last.0 - prev.0;
                let dir_y = last.1 - prev.1;
                buffer.draw_sheet_arrow(last.0, last.1, last.2, dir_x, dir_y, arrow_length, arrow_width, last.3);
            }
        }
    }

    /// Draw a ribbon segment between two points
    fn draw_ribbon(
        &self,
        buffer: &mut PixelBuffer,
        x1: f32,
        y1: f32,
        z1: f32,
        x2: f32,
        y2: f32,
        z2: f32,
        color: (u8, u8, u8),
        width: f32,
    ) {
        // Calculate perpendicular direction for ribbon width
        let dx = x2 - x1;
        let dy = y2 - y1;
        let len = (dx * dx + dy * dy).sqrt();

        if len < 0.001 {
            return;
        }

        // Normalize and rotate 90 degrees to get perpendicular
        let perpx = -dy / len;
        let perpy = dx / len;

        // Draw multiple parallel lines to create ribbon effect
        let half_width = (width * 0.5).max(1.0);
        let steps = (half_width as i32).max(1);

        for i in -steps..=steps {
            let offset = (i as f32) * 0.5;
            let ox = perpx * offset;
            let oy = perpy * offset;

            buffer.draw_line(
                x1 + ox,
                y1 + oy,
                z1,
                x2 + ox,
                y2 + oy,
                z2,
                color,
            );
        }
    }

    /// Draw a smooth spline-based ribbon segment using Catmull-Rom interpolation
    fn draw_spline_ribbon(
        &self,
        buffer: &mut PixelBuffer,
        p0: (f32, f32, f32),  // Control point before start
        p1: (f32, f32, f32),  // Start point
        p2: (f32, f32, f32),  // End point
        p3: (f32, f32, f32),  // Control point after end
        color: (u8, u8, u8),
        width: f32,
    ) {
        // Number of interpolation steps (more = smoother)
        let steps = 8;

        let mut prev_x = p1.0;
        let mut prev_y = p1.1;
        let mut prev_z = p1.2;

        for i in 1..=steps {
            let t = i as f32 / steps as f32;

            // Catmull-Rom spline interpolation
            let t2 = t * t;
            let t3 = t2 * t;

            let x = 0.5 * ((2.0 * p1.0)
                + (-p0.0 + p2.0) * t
                + (2.0 * p0.0 - 5.0 * p1.0 + 4.0 * p2.0 - p3.0) * t2
                + (-p0.0 + 3.0 * p1.0 - 3.0 * p2.0 + p3.0) * t3);

            let y = 0.5 * ((2.0 * p1.1)
                + (-p0.1 + p2.1) * t
                + (2.0 * p0.1 - 5.0 * p1.1 + 4.0 * p2.1 - p3.1) * t2
                + (-p0.1 + 3.0 * p1.1 - 3.0 * p2.1 + p3.1) * t3);

            let z = 0.5 * ((2.0 * p1.2)
                + (-p0.2 + p2.2) * t
                + (2.0 * p0.2 - 5.0 * p1.2 + 4.0 * p2.2 - p3.2) * t2
                + (-p0.2 + 3.0 * p1.2 - 3.0 * p2.2 + p3.2) * t3);

            // Draw ribbon segment
            self.draw_ribbon(buffer, prev_x, prev_y, prev_z, x, y, z, color, width);

            prev_x = x;
            prev_y = y;
            prev_z = z;
        }
    }

    /// Draw a shaded ribbon segment with 3D tube appearance
    fn draw_ribbon_shaded(
        &self,
        buffer: &mut PixelBuffer,
        x1: f32,
        y1: f32,
        z1: f32,
        x2: f32,
        y2: f32,
        z2: f32,
        color: (u8, u8, u8),
        width: f32,
    ) {
        buffer.draw_cylinder_shaded(x1, y1, z1, x2, y2, z2, width * 0.5, color);
    }

    /// Draw a smooth spline-based ribbon with shading (for helices/coils)
    fn draw_spline_ribbon_shaded(
        &self,
        buffer: &mut PixelBuffer,
        p0: (f32, f32, f32),
        p1: (f32, f32, f32),
        p2: (f32, f32, f32),
        p3: (f32, f32, f32),
        color: (u8, u8, u8),
        width: f32,
    ) {
        let steps = 6;
        let mut prev_x = p1.0;
        let mut prev_y = p1.1;
        let mut prev_z = p1.2;

        for i in 1..=steps {
            let t = i as f32 / steps as f32;
            let t2 = t * t;
            let t3 = t2 * t;

            let x = 0.5 * ((2.0 * p1.0)
                + (-p0.0 + p2.0) * t
                + (2.0 * p0.0 - 5.0 * p1.0 + 4.0 * p2.0 - p3.0) * t2
                + (-p0.0 + 3.0 * p1.0 - 3.0 * p2.0 + p3.0) * t3);

            let y = 0.5 * ((2.0 * p1.1)
                + (-p0.1 + p2.1) * t
                + (2.0 * p0.1 - 5.0 * p1.1 + 4.0 * p2.1 - p3.1) * t2
                + (-p0.1 + 3.0 * p1.1 - 3.0 * p2.1 + p3.1) * t3);

            let z = 0.5 * ((2.0 * p1.2)
                + (-p0.2 + p2.2) * t
                + (2.0 * p0.2 - 5.0 * p1.2 + 4.0 * p2.2 - p3.2) * t2
                + (-p0.2 + 3.0 * p1.2 - 3.0 * p2.2 + p3.2) * t3);

            self.draw_ribbon_shaded(buffer, prev_x, prev_y, prev_z, x, y, z, color, width);

            prev_x = x;
            prev_y = y;
            prev_z = z;
        }
    }

    /// Draw a smooth spline-based flat sheet (for beta strands)
    fn draw_spline_sheet_shaded(
        &self,
        buffer: &mut PixelBuffer,
        p0: (f32, f32, f32),
        p1: (f32, f32, f32),
        p2: (f32, f32, f32),
        p3: (f32, f32, f32),
        color: (u8, u8, u8),
        width: f32,
    ) {
        let steps = 6;
        let mut prev_x = p1.0;
        let mut prev_y = p1.1;
        let mut prev_z = p1.2;

        for i in 1..=steps {
            let t = i as f32 / steps as f32;
            let t2 = t * t;
            let t3 = t2 * t;

            let x = 0.5 * ((2.0 * p1.0)
                + (-p0.0 + p2.0) * t
                + (2.0 * p0.0 - 5.0 * p1.0 + 4.0 * p2.0 - p3.0) * t2
                + (-p0.0 + 3.0 * p1.0 - 3.0 * p2.0 + p3.0) * t3);

            let y = 0.5 * ((2.0 * p1.1)
                + (-p0.1 + p2.1) * t
                + (2.0 * p0.1 - 5.0 * p1.1 + 4.0 * p2.1 - p3.1) * t2
                + (-p0.1 + 3.0 * p1.1 - 3.0 * p2.1 + p3.1) * t3);

            let z = 0.5 * ((2.0 * p1.2)
                + (-p0.2 + p2.2) * t
                + (2.0 * p0.2 - 5.0 * p1.2 + 4.0 * p2.2 - p3.2) * t2
                + (-p0.2 + 3.0 * p1.2 - 3.0 * p2.2 + p3.2) * t3);

            // Use flat sheet instead of cylinder for beta strands
            buffer.draw_flat_sheet(prev_x, prev_y, prev_z, x, y, z, width, color);

            prev_x = x;
            prev_y = y;
            prev_z = z;
        }
    }

    fn draw_arrowhead(
        &self,
        buffer: &mut PixelBuffer,
        tip_x: f32,
        tip_y: f32,
        tip_z: f32,
        dir: Vector2<f32>,
        color: (u8, u8, u8),
        length: f32,
        width: f32,
    ) {
        let dir_len = (dir.x * dir.x + dir.y * dir.y).sqrt();
        if dir_len < 1e-3 {
            return;
        }

        let dx = dir.x / dir_len;
        let dy = dir.y / dir_len;
        let px = -dy;
        let py = dx;

        let length = length.max(2.0);
        let width = width.max(2.0);
        let base_x = tip_x - dx * length;
        let base_y = tip_y - dy * length;
        let steps = (length.round() as i32).clamp(2, 12);

        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let cx = base_x + dx * length * t;
            let cy = base_y + dy * length * t;
            let half = (width * t * 0.5).max(0.5);
            let ox = px * half;
            let oy = py * half;
            buffer.draw_line(cx - ox, cy - oy, tip_z, cx + ox, cy + oy, tip_z, color);
        }
    }

    fn current_assembly(&self) -> &Assembly {
        if self.molecule.assemblies.is_empty() {
            &self.default_assembly
        } else {
            &self.molecule.assemblies[self.assembly_index]
        }
    }

    fn current_assembly_id(&self) -> &str {
        if self.molecule.assemblies.is_empty() {
            "1"
        } else {
            self.molecule.assemblies[self.assembly_index].id.as_str()
        }
    }

    fn current_assembly_bounds(&self) -> (Vector3<f32>, Vector3<f32>) {
        let mut min = Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        let mut has_any = false;

        for instance in &self.current_assembly().instances {
            for atom in &self.molecule.atoms {
                if !instance.applies_to_chain(atom.chain_id) {
                    continue;
                }
                let coord = instance.transform.apply(atom.coord);
                if !has_any {
                    min = coord;
                    max = coord;
                    has_any = true;
                } else {
                    min.x = min.x.min(coord.x);
                    min.y = min.y.min(coord.y);
                    min.z = min.z.min(coord.z);
                    max.x = max.x.max(coord.x);
                    max.y = max.y.max(coord.y);
                    max.z = max.z.max(coord.z);
                }
            }
        }

        if has_any {
            (min, max)
        } else {
            (Vector3::zeros(), Vector3::zeros())
        }
    }

    /// Cycle through available render backends
    fn cycle_backend(&mut self) {
        use output::RenderBackend;
        self.backend = match self.backend {
            RenderBackend::HalfBlock => RenderBackend::ITerm2,
            RenderBackend::ITerm2 => RenderBackend::HalfBlock,
        };
        // Update image interval for new protocol
        self.image_interval = match self.backend {
            RenderBackend::ITerm2 => Duration::from_millis(50), // ~20 FPS
            RenderBackend::HalfBlock => Duration::from_millis(16), // ~60 FPS
        };
        self.backend_changed = true;
    }

    fn fit_camera_to_current_assembly(&mut self) {
        let (min, max) = self.current_assembly_bounds();
        let center = (min + max) / 2.0;
        self.camera.reset(center);
        let size = max - min;
        let max_dim = size.x.max(size.y).max(size.z);
        if max_dim > 0.0 {
            self.camera.zoom = 2.0 / max_dim;
        }
    }

    /// Compute the screen-space bounding box of all visible atoms.
    /// Returns (min_x, min_y, max_x, max_y) in pixel coordinates.
    /// Also accounts for atom radii based on representation.
    fn compute_projected_bounds(&self, width: usize, height: usize) -> Option<(f32, f32, f32, f32)> {
        let pixel_width = width as f32;
        let pixel_height = height as f32;
        let center_x = pixel_width / 2.0;
        let center_y = pixel_height / 2.0;
        let min_dim = pixel_width.min(pixel_height);
        let base_scale = min_dim * 0.45;
        let screen_scale = min_dim * 0.5;

        // Base radius for atoms/elements in screen space
        // This depends on representation - cartoon tubes are wider than backbone lines
        let base_radius = match self.representation {
            Representation::Cartoon => base_scale * 0.04,
            Representation::Surface => base_scale * 0.05,
            Representation::Backbone => base_scale * 0.01,
        };

        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        let radius_scale = base_radius / self.camera.zoom;

        for instance in &self.current_assembly().instances {
            for atom in &self.molecule.atoms {
                if !self.is_atom_visible(atom) || !instance.applies_to_chain(atom.chain_id) {
                    continue;
                }

                let world = instance.transform.apply(atom.coord);
                let (screen_pos, _z, size_scale) = self.camera.project_with_scale(world, screen_scale);
                let sx = center_x + screen_pos.x * base_scale;
                let sy = center_y + screen_pos.y * base_scale;

                let effective_radius = radius_scale * size_scale;

                min_x = min_x.min(sx - effective_radius);
                min_y = min_y.min(sy - effective_radius);
                max_x = max_x.max(sx + effective_radius);
                max_y = max_y.max(sy + effective_radius);
            }
        }

        if min_x <= max_x && min_y <= max_y {
            Some((min_x, min_y, max_x, max_y))
        } else {
            None
        }
    }

    /// Adjust camera zoom so the molecule fills the frame with a small margin.
    /// This is a two-pass approach: project once to measure, then adjust zoom.
    fn fit_zoom_to_frame(&mut self, width: usize, height: usize, margin: f32) {
        let Some((min_x, min_y, max_x, max_y)) = self.compute_projected_bounds(width, height) else {
            return;
        };

        let pixel_width = width as f32;
        let pixel_height = height as f32;
        let center_x = pixel_width / 2.0;
        let center_y = pixel_height / 2.0;

        // Calculate current extent from center (max distance to any edge, doubled for full span)
        let used_width = (max_x - center_x).max(center_x - min_x) * 2.0;
        let used_height = (max_y - center_y).max(center_y - min_y) * 2.0;

        if used_width < 1.0 || used_height < 1.0 {
            return;
        }

        // Calculate scale to fill the frame with margin, preserving aspect ratio
        let target_width = pixel_width - margin * 2.0;
        let target_height = pixel_height - margin * 2.0;
        let scale = (target_width / used_width)
            .min(target_height / used_height)
            .clamp(0.5, 2.0);

        self.camera.zoom *= scale;
    }
}

/// Run benchmark mode: high-quality rendering with auto-rotation for 2 seconds
/// Outputs detailed performance metrics to benchmark.log
/// Works headless (no terminal required) for CI/automation
pub fn run_benchmark(path: &Path, molecule: Molecule) -> Result<(), UiError> {
    use std::fs::File;
    use std::io::Write as IoWrite;

    const BENCHMARK_DURATION: Duration = Duration::from_secs(2);
    const LOG_FILE: &str = "benchmark.log";

    // Headless benchmark - no terminal needed
    let render_backend = RenderBackend::HalfBlock; // Use half-block for headless
    let mut app = App::new(path, molecule, render_backend);

    // Enable high quality settings for benchmark
    app.shading_enabled = true;
    app.auto_spin = true;
    app.show_hud = false;
    app.representation = crate::render::Representation::Cartoon;

    // Fixed resolution for reproducible benchmarks (simulates 160x48 terminal with 2x4 pixels/cell)
    let mol_width: u16 = 160;
    let mol_height: u16 = 48;
    let pixels_per_cell = (2, 4); // High quality
    let render_width = mol_width as usize * pixels_per_cell.0;
    let render_height = mol_height as usize * pixels_per_cell.1;

    println!("Starting benchmark...");
    println!("  File: {}", path.display());
    println!("  Atoms: {}, Bonds: {}", app.molecule.atoms.len(), app.molecule.bonds.len());
    println!("  Resolution: {}x{}", render_width, render_height);
    println!("  Shading: {}, Representation: {:?}", app.shading_enabled, app.representation);

    // Collect timing data
    let mut frame_times: Vec<f64> = Vec::new();
    let mut render_times: Vec<f64> = Vec::new();
    let mut post_times: Vec<f64> = Vec::new();
    let mut clear_times: Vec<f64> = Vec::new();

    let benchmark_start = Instant::now();
    let mut buffer = PixelBuffer::new(render_width, render_height);
    let mut total_frames = 0u64;

    // Benchmark loop
    while benchmark_start.elapsed() < BENCHMARK_DURATION {
        let frame_start = Instant::now();

        // Auto-spin (faster for benchmark to stress test)
        let rotation_delta = Vector2::new(0.02, 0.005);
        app.camera.trackball_rotate(Vector2::zeros(), rotation_delta);

        let t0 = Instant::now();
        buffer.resize_or_clear(render_width, render_height);
        let t1 = Instant::now();

        app.render_molecule(&mut buffer, pixels_per_cell);
        let t2 = Instant::now();

        // Apply post-processing (same as real rendering)
        if app.representation == Representation::Surface {
            for _ in 0..5 {
                fill_depth_gaps(&mut buffer, 3, f32::INFINITY);
            }
        }
        apply_edge_aa(&mut buffer, 0.55, 0.06);
        apply_silhouette_edges(&mut buffer, 0.15, 0.5);
        let t3 = Instant::now();

        // Record timings (in milliseconds)
        clear_times.push(t1.duration_since(t0).as_secs_f64() * 1000.0);
        render_times.push(t2.duration_since(t1).as_secs_f64() * 1000.0);
        post_times.push(t3.duration_since(t2).as_secs_f64() * 1000.0);
        frame_times.push(frame_start.elapsed().as_secs_f64() * 1000.0);

        total_frames += 1;

        // Progress indicator
        if total_frames % 10 == 0 {
            print!(".");
            std::io::stdout().flush().ok();
        }
    }
    println!();

    let total_duration = benchmark_start.elapsed();

    // Calculate statistics
    let avg = |v: &[f64]| -> f64 { v.iter().sum::<f64>() / v.len().max(1) as f64 };
    let min = |v: &[f64]| -> f64 { v.iter().cloned().fold(f64::INFINITY, f64::min) };
    let max = |v: &[f64]| -> f64 { v.iter().cloned().fold(f64::NEG_INFINITY, f64::max) };
    let percentile = |v: &mut [f64], p: f64| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((p / 100.0) * (v.len() - 1) as f64) as usize;
        v.get(idx).copied().unwrap_or(0.0)
    };

    let avg_fps = total_frames as f64 / total_duration.as_secs_f64();
    let avg_frame = avg(&frame_times);
    let min_frame = min(&frame_times);
    let max_frame = max(&frame_times);
    let mut frame_times_sorted = frame_times.clone();
    let p95_frame = percentile(&mut frame_times_sorted, 95.0);
    let p99_frame = percentile(&mut frame_times_sorted, 99.0);

    // Write to log file
    let mut log = File::create(LOG_FILE).map_err(|e| UiError::TerminalError(e))?;

    writeln!(log, "=== PDBCAT BENCHMARK RESULTS ===").ok();
    writeln!(log, "File: {}", path.display()).ok();
    writeln!(log, "Backend: {} (headless)", render_backend.label()).ok();
    writeln!(log, "Resolution: {}x{}", render_width, render_height).ok();
    writeln!(log, "Representation: {:?}", app.representation).ok();
    writeln!(log, "Shading: {}", app.shading_enabled).ok();
    writeln!(log, "Atoms: {}", app.molecule.atoms.len()).ok();
    writeln!(log, "Bonds: {}", app.molecule.bonds.len()).ok();
    writeln!(log, "").ok();
    writeln!(log, "=== SUMMARY ===").ok();
    writeln!(log, "Duration: {:.2}s", total_duration.as_secs_f64()).ok();
    writeln!(log, "Total frames: {}", total_frames).ok();
    writeln!(log, "Average FPS: {:.1}", avg_fps).ok();
    writeln!(log, "").ok();
    writeln!(log, "=== FRAME TIME (ms) ===").ok();
    writeln!(log, "  avg: {:.2}", avg_frame).ok();
    writeln!(log, "  min: {:.2}", min_frame).ok();
    writeln!(log, "  max: {:.2}", max_frame).ok();
    writeln!(log, "  p95: {:.2}", p95_frame).ok();
    writeln!(log, "  p99: {:.2}", p99_frame).ok();
    writeln!(log, "").ok();
    writeln!(log, "=== BREAKDOWN (ms, avg) ===").ok();
    writeln!(log, "  clear:      {:.2}", avg(&clear_times)).ok();
    writeln!(log, "  render:     {:.2}", avg(&render_times)).ok();
    writeln!(log, "  post:       {:.2}", avg(&post_times)).ok();
    writeln!(log, "").ok();
    writeln!(log, "=== PER-FRAME DATA ===").ok();
    writeln!(log, "frame,total_ms,clear_ms,render_ms,post_ms").ok();
    for i in 0..total_frames as usize {
        writeln!(log, "{},{:.3},{:.3},{:.3},{:.3}",
            i,
            frame_times.get(i).unwrap_or(&0.0),
            clear_times.get(i).unwrap_or(&0.0),
            render_times.get(i).unwrap_or(&0.0),
            post_times.get(i).unwrap_or(&0.0),
        ).ok();
    }

    // Print summary to stdout
    println!("\n=== BENCHMARK RESULTS ===");
    println!("Duration: {:.2}s, Frames: {}, Avg FPS: {:.1}",
        total_duration.as_secs_f64(), total_frames, avg_fps);
    println!("Frame time: avg={:.1}ms, min={:.1}ms, max={:.1}ms, p95={:.1}ms, p99={:.1}ms",
        avg_frame, min_frame, max_frame, p95_frame, p99_frame);
    println!("Breakdown: clear={:.2}ms, render={:.2}ms, post={:.2}ms",
        avg(&clear_times), avg(&render_times), avg(&post_times));
    println!("Detailed results written to: {}", LOG_FILE);

    Ok(())
}

/// Render molecule to a PNG file without interactive viewer
pub fn render_to_png(
    path: &Path,
    molecule: Molecule,
    output_path: &Path,
    width: usize,
    height: usize,
    options: RenderOptions,
) -> Result<(), UiError> {
    use std::fs::File;
    use std::io::Write;

    // Create a minimal app state for rendering
    let render_backend = RenderBackend::HalfBlock; // Doesn't matter for PNG output
    let mut app = App::new(path, molecule, render_backend);
    app.representation = options.representation;
    app.color_scheme = options.color_scheme;
    app.shading_enabled = options.shading;

    // Auto-fit zoom for non-interactive mode (fills frame with small margin)
    app.fit_zoom_to_frame(width, height, 4.0);

    // Create pixel buffer at desired resolution
    let mut buffer = PixelBuffer::new(width, height);
    if let Some(bg) = options.background {
        buffer.fill_background(bg);
    }

    // Use 1:1 pixel mapping (no cell subdivision)
    let pixels_per_cell = (1, 1);

    // Render the molecule
    app.render_molecule(&mut buffer, pixels_per_cell);

    // Apply post-processing for professional look
    if options.shading {
        // Fill gaps in Surface representation to fix concave "ray" artifacts
        // Multiple passes with increasing radii to fill larger gaps iteratively
        if options.representation == Representation::Surface {
            for _ in 0..5 {
                fill_depth_gaps(&mut buffer, 3, f32::INFINITY);
            }
        }
        apply_ssao(&mut buffer, 5.0, 0.5); // Subtle SSAO for depth
        apply_edge_aa(&mut buffer, 0.25, 0.02); // Soft edge AA
        apply_silhouette_edges(&mut buffer, 0.04, 0.15); // Very subtle outline
        apply_tone_mapping(&mut buffer, 1.05);
    }

    // Encode to PNG
    let png_data = output::encode_png_rgb(&buffer);

    // Write to file
    let mut file = File::create(output_path)?;
    file.write_all(&png_data)?;

    Ok(())
}

/// Render molecule to stdout (auto-detect backend)
/// For iTerm2: outputs high-res inline image
/// For other terminals: outputs half-block characters
pub fn render_to_stdout(
    path: &Path,
    molecule: Molecule,
    resolution: Option<(usize, usize)>,
    options: RenderOptions,
) -> Result<(), UiError> {
    use std::io::Write;

    // Use override backend if provided, otherwise auto-detect
    let backend = options.backend.unwrap_or_else(detect_backend);

    // Get terminal size for default resolution
    let (term_cols, term_rows) = terminal::size().unwrap_or((80, 24));

    let (width, height) = match resolution {
        Some((w, h)) => (w, h),
        None => match backend {
            // iTerm2: high-res image based on terminal size
            // Assume ~10 pixels per character cell for good quality
            RenderBackend::ITerm2 => {
                let w = (term_cols as usize) * 10;
                let h = (term_rows as usize) * 20; // ~2:1 aspect for terminal cells
                (w.min(1920), h.min(1080)) // Cap at reasonable max
            }
            // Half-block: 1 char = 1x2 pixels
            RenderBackend::HalfBlock => {
                (term_cols as usize, (term_rows as usize) * 2)
            }
        }
    };

    // Create app state for rendering
    let render_backend = RenderBackend::HalfBlock;
    let mut app = App::new(path, molecule, render_backend);
    app.representation = options.representation;
    app.color_scheme = options.color_scheme;
    app.shading_enabled = options.shading;

    // Auto-fit zoom for non-interactive mode (fills frame with small margin)
    app.fit_zoom_to_frame(width, height, 4.0);

    // Create pixel buffer at desired resolution
    let mut buffer = PixelBuffer::new(width, height);
    if let Some(bg) = options.background {
        buffer.fill_background(bg);
    }

    // Use 1:1 pixel mapping
    let pixels_per_cell = (1, 1);

    // Render the molecule
    app.render_molecule(&mut buffer, pixels_per_cell);

    // Apply post-processing
    if options.shading {
        if options.representation == Representation::Surface {
            for _ in 0..5 {
                fill_depth_gaps(&mut buffer, 3, f32::INFINITY);
            }
        }
        apply_ssao(&mut buffer, 5.0, 0.5);
        apply_edge_aa(&mut buffer, 0.25, 0.02);
        apply_silhouette_edges(&mut buffer, 0.04, 0.15);
        apply_tone_mapping(&mut buffer, 1.05);
    }

    // Output based on backend
    let mut stdout = io::stdout();
    match backend {
        RenderBackend::ITerm2 => {
            // Output iTerm2 inline image
            output::render_iterm2_image(&buffer, term_cols, term_rows, &mut stdout)?;
            stdout.write_all(b"\n")?;
        }
        RenderBackend::HalfBlock => {
            // Output half-block characters
            crate::render::braille::render_half_block(&buffer, &mut stdout)?;
        }
    }
    stdout.flush()?;

    Ok(())
}

pub fn run(path: &Path, molecule: Molecule) -> Result<(), UiError> {
    // Setup terminal
    terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        terminal::EnterAlternateScreen,
        cursor::Hide,
        event::EnableMouseCapture
    )?;

    let crossterm_backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(crossterm_backend)?;

    let render_backend = detect_backend();
    let mut app = App::new(path, molecule, render_backend);

    // Main loop
    let result = run_loop(&mut terminal, &mut app);

    // Cleanup
    terminal::disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        terminal::LeaveAlternateScreen,
        cursor::Show,
        event::DisableMouseCapture
    )?;

    result
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &mut App,
) -> Result<(), UiError> {
    // Reusable pixel buffer - avoids allocation each frame
    let mut buffer = PixelBuffer::new(1, 1);
    let mut last_size = (0u16, 0u16);

    loop {
        let now = Instant::now();

        // Auto-spin (slow rotation for viewing)
        if app.auto_spin {
            let rotation_delta = Vector2::new(0.0013, 0.0);
            app.camera.trackball_rotate(Vector2::zeros(), rotation_delta);
            app.last_interaction = now; // Mark as active during spin
            app.needs_redraw = true;
        }

        let size = terminal.size()?;

        // Check if terminal size changed
        if (size.width, size.height) != last_size {
            last_size = (size.width, size.height);
            app.needs_redraw = true;
        }

        // Skip rendering if nothing changed
        if !app.needs_redraw && !app.backend_changed {
            // Longer timeout when idle
            let timeout = Duration::from_millis(100);
            if event::poll(timeout)? {
                match event::read()? {
                    Event::Key(key) if key.kind == event::KeyEventKind::Press => {
                        app.handle_key(key.code, key.modifiers);
                    }
                    Event::Mouse(mouse) => {
                        app.handle_mouse(mouse, size.width, size.height);
                    }
                    Event::Resize(_, _) => {
                        app.needs_redraw = true;
                    }
                    _ => {}
                }
            }
            if app.should_quit {
                break;
            }
            continue;
        }

        let mut mol_height = if app.show_hud {
            size.height.saturating_sub(1) // Reserve 1 line for HUD at bottom
        } else {
            size.height
        };
        if mol_height == 0 {
            mol_height = 1;
        }
        let mol_width = size.width.max(1);

        // Adaptive resolution: high quality when idle, lower during motion
        let idle_duration = now.duration_since(app.last_interaction);
        let is_idle = idle_duration > Duration::from_millis(300);

        let image_active = matches!(app.backend, RenderBackend::ITerm2) && !app.show_help;
        let pixels_per_cell = if app.show_help {
            (1, 2)
        } else {
            match app.backend {
                RenderBackend::HalfBlock => (1, 2),
                // iTerm2 uses PNG - higher res when idle
                RenderBackend::ITerm2 => {
                    if is_idle { (3, 6) } else { (2, 4) }
                }
            }
        };
        let pixel_width = mol_width as usize * pixels_per_cell.0;
        let pixel_height = mol_height as usize * pixels_per_cell.1;

        let render_width = pixel_width;
        let render_height = pixel_height;
        let render_pixels_per_cell = pixels_per_cell;

        // Clear screen when backend changes to remove stale images from previous protocol
        if app.backend_changed {
            execute!(
                terminal.backend_mut(),
                terminal::Clear(terminal::ClearType::All),
                cursor::MoveTo(0, 0)
            )?;
            app.backend_changed = false;
        }

        // Resize buffer only if dimensions changed, otherwise just clear
        buffer.resize_or_clear(render_width, render_height);

        app.render_molecule(&mut buffer, render_pixels_per_cell);

        // Apply post-processing for professional look
        if app.shading_enabled {
            // Fill gaps in Surface representation to fix concave "ray" artifacts
            // Multiple passes to fill larger gaps iteratively
            if app.representation == Representation::Surface {
                for _ in 0..3 {
                    fill_depth_gaps(&mut buffer, 2, f32::INFINITY);
                }
            }

            // SSAO for depth and contact shadows (subtle but adds depth)
            apply_ssao(&mut buffer, 8.0, 1.0);

            // Edge AA for smoother sphere/cylinder edges
            if image_active {
                apply_edge_aa(&mut buffer, 0.55, 0.06);
            }

            // Silhouette edges for ChimeraX-style outlines
            apply_silhouette_edges(&mut buffer, 0.12, 0.5);

            // Tone mapping for better color reproduction and contrast
            apply_tone_mapping(&mut buffer, 1.15);
        }

        let final_buffer: &PixelBuffer = &buffer;

        // Render TUI
        terminal.draw(|f| {
            let buffer_ref = if image_active { None } else { Some(final_buffer) };
            draw_ui(f, app, app.backend, mol_height, buffer_ref);
        })?;

        // Send image for iTerm2 backend
        if image_active && matches!(app.backend, RenderBackend::ITerm2) {
            output::render_iterm2_image(final_buffer, mol_width, mol_height, terminal.backend_mut())?;
            app.last_image_sent = now;
        }

        // Mark as drawn
        app.needs_redraw = false;

        // Handle events with short timeout for responsive input
        let timeout = if app.auto_spin {
            Duration::from_millis(16) // ~60 FPS for smooth animation
        } else {
            Duration::from_millis(100)
        };

        if event::poll(timeout)? {
            match event::read()? {
                Event::Key(key) if key.kind == event::KeyEventKind::Press => {
                    app.handle_key(key.code, key.modifiers);
                }
                Event::Mouse(mouse) => {
                    app.handle_mouse(mouse, size.width, size.height);
                }
                Event::Resize(_, _) => {
                    app.needs_redraw = true;
                }
                _ => {}
            }
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

fn draw_ui(
    f: &mut Frame,
    app: &mut App,
    backend: RenderBackend,
    mol_height: u16,
    buffer: Option<&PixelBuffer>,
) {
    let size = f.size();
    let mol_height = mol_height.min(size.height);
    let mol_area = Rect::new(0, 0, size.width, mol_height);

    if let Some(buffer) = buffer {
        let lines = render_half_block_lines(buffer);
        let paragraph = Paragraph::new(lines);
        f.render_widget(paragraph, mol_area);
    } else {
        f.render_widget(Clear, mol_area);
    }

    if app.show_hud {
        let hud_height = size.height.saturating_sub(mol_height);
        if hud_height > 0 {
            let hud_area = Rect::new(0, mol_height, size.width, hud_height);
            let hud_text = format!(
                " {} | {} chains, {} atoms | {} | {} | {} | Backend: {} | Asm: {} ({}/{}) | AltLoc: {} | q: quit, h: help, F1: HUD",
                app.filename,
                app.molecule.chain_count(),
                app.molecule.atom_count(),
                app.representation.name(),
                app.color_scheme.name(),
                app.camera.projection.name(),
                backend.label(),
                app.current_assembly_id(),
                app.assembly_index + 1,
                app.molecule.assembly_count(),
                app.alt_loc_mode.label(),
            );
            let hud = Paragraph::new(hud_text)
                .style(Style::default().fg(Color::White).bg(Color::DarkGray))
                .block(Block::default());
            f.render_widget(hud, hud_area);
        }
    }

    if app.show_help {
        render_help_overlay(f);
    }
}

const UPPER_HALF_BLOCK: char = '\u{2580}';
const LOWER_HALF_BLOCK: char = '\u{2584}';

fn render_half_block_lines(buffer: &PixelBuffer) -> Vec<Line<'_>> {
    let cell_width = buffer.width();
    let cell_height = buffer.height() / 2;
    let mut lines = Vec::with_capacity(cell_height);

    for cell_y in 0..cell_height {
        let mut spans: Vec<Span> = Vec::with_capacity(cell_width);
        for cell_x in 0..cell_width {
            let px = cell_x;
            let py = cell_y * 2;
            let top = sample_region(buffer, px, py, 1, 1);
            let bottom = sample_region(buffer, px, py + 1, 1, 1);

            let (ch, style) = match (top, bottom) {
                (Some(top), Some(bottom)) => (
                    UPPER_HALF_BLOCK,
                    Style::default()
                        .fg(Color::Rgb(top.0, top.1, top.2))
                        .bg(Color::Rgb(bottom.0, bottom.1, bottom.2)),
                ),
                (Some(top), None) => (
                    UPPER_HALF_BLOCK,
                    Style::default().fg(Color::Rgb(top.0, top.1, top.2)).bg(Color::Reset),
                ),
                (None, Some(bottom)) => (
                    LOWER_HALF_BLOCK,
                    Style::default().fg(Color::Rgb(bottom.0, bottom.1, bottom.2)).bg(Color::Reset),
                ),
                (None, None) => (' ', Style::default()),
            };

            spans.push(Span::styled(ch.to_string(), style));
        }
        lines.push(Line::from(spans));
    }

    lines
}

fn sample_region(
    buffer: &PixelBuffer,
    start_x: usize,
    start_y: usize,
    width: usize,
    height: usize,
) -> Option<(u8, u8, u8)> {
    let mut sum_r: u32 = 0;
    let mut sum_g: u32 = 0;
    let mut sum_b: u32 = 0;
    let mut count: u32 = 0;

    for y in start_y..start_y + height {
        for x in start_x..start_x + width {
            let (r, g, b, a) = buffer.get_pixel(x, y);
            if a > 0 {
                sum_r += r as u32;
                sum_g += g as u32;
                sum_b += b as u32;
                count += 1;
            }
        }
    }

    if count == 0 {
        None
    } else {
        Some((
            (sum_r / count) as u8,
            (sum_g / count) as u8,
            (sum_b / count) as u8,
        ))
    }
}

const HELP_LINES: &[&str] = &[
    "View",
    "  Mouse drag     : rotate",
    "  Scroll / [ ]   : zoom",
    "  Arrow keys     : pan",
    "  Shift+Arrows   : rotate",
    "  0              : reset view",
    "  v              : toggle projection",
    "  p              : toggle auto-spin",
    "",
    "Display",
    "  Tab            : cycle representation",
    "  c              : cycle color scheme",
    "  s              : toggle shading",
    "  b              : cycle backend",
    "  /              : cycle assemblies",
    "  '              : cycle altloc",
    "  l              : toggle ligands",
    "  F3             : toggle waters",
    "  A-Z            : toggle chain visibility",
    "  F1             : toggle HUD",
    "  h              : toggle help",
    "",
    "Exit",
    "  q or Esc       : quit",
];

fn render_help_overlay(f: &mut Frame) {
    let size = f.size();
    let max_len = HELP_LINES
        .iter()
        .map(|line| line.len())
        .max()
        .unwrap_or(0);
    let max_width = size.width.max(1);
    let max_height = size.height.max(1);
    let content_width = (max_len as u16).min(max_width.saturating_sub(2));
    let content_height = (HELP_LINES.len() as u16).min(max_height.saturating_sub(2));

    let box_width = (content_width + 2).max(10).min(max_width);
    let box_height = (content_height + 2).max(6).min(max_height);

    let x = size.width.saturating_sub(box_width) / 2;
    let y = size.height.saturating_sub(box_height) / 2;
    let area = Rect::new(x, y, box_width, box_height);

    let lines: Vec<Line> = HELP_LINES.iter().map(|line| Line::from(*line)).collect();
    let help = Paragraph::new(lines)
        .block(
            Block::default()
                .title("Help")
                .borders(Borders::ALL)
                .style(Style::default().fg(Color::White).bg(Color::Black)),
        );

    f.render_widget(Clear, area);
    f.render_widget(help, area);
}
