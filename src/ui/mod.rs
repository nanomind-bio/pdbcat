//! Terminal UI and application loop
//!
//! Handles keyboard/mouse input, rendering to terminal, and HUD display.

mod output;

use crate::molecule::{Assembly, Molecule, SecondaryStructure};
use crate::render::{PixelBuffer, Camera, ColorScheme, Representation, chain_color, rainbow_color};
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
use std::io::{self, Stdout};
use std::path::Path;
use std::time::{Duration, Instant};
use thiserror::Error;
use nalgebra::{Vector2, Vector3};
use output::{RenderBackend, detect_backend, render_image};

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
    /// Chain visibility (indexed by chain ID - 'A' offset)
    chain_visible: [bool; 26],
    /// Mouse drag state
    mouse_drag: Option<(u16, u16)>,
    /// Last frame time for FPS calculation
    last_frame: Instant,
    /// Current FPS
    fps: f32,
    /// Whether to quit
    should_quit: bool,
}

impl App {
    fn new(path: &Path, molecule: Molecule) -> Self {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let center = molecule.center();
        let camera = Camera::new(center);

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
            shading_enabled: false,
            auto_spin: false,
            alt_loc_mode: AltLocMode::A,
            default_assembly: Assembly::single_identity(),
            assembly_index: 0,
            chain_visible: [true; 26],
            mouse_drag: None,
            last_frame: Instant::now(),
            fps: 0.0,
            should_quit: false,
        };

        app.fit_camera_to_current_assembly();
        app
    }

    fn handle_key(&mut self, code: KeyCode, modifiers: KeyModifiers) {
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
            KeyCode::Char('0') => {
                self.fit_camera_to_current_assembly();
            }
            KeyCode::Up => {
                if modifiers.contains(KeyModifiers::SHIFT) {
                    self.camera.zoom_by(1.1);
                } else {
                    self.camera.pan(Vector2::new(0.0, -0.1));
                }
            }
            KeyCode::Down => {
                if modifiers.contains(KeyModifiers::SHIFT) {
                    self.camera.zoom_by(0.9);
                } else {
                    self.camera.pan(Vector2::new(0.0, 0.1));
                }
            }
            KeyCode::Left => {
                self.camera.pan(Vector2::new(-0.1, 0.0));
            }
            KeyCode::Right => {
                self.camera.pan(Vector2::new(0.1, 0.0));
            }
            KeyCode::Char(c) if c.is_ascii_uppercase() => {
                let idx = (c as u8 - b'A') as usize;
                if idx < 26 {
                    self.chain_visible[idx] = !self.chain_visible[idx];
                }
            }
            _ => {}
        }
    }

    fn handle_mouse(&mut self, event: MouseEvent, width: u16, height: u16) {
        match event.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                self.mouse_drag = Some((event.column, event.row));
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                if let Some((prev_x, prev_y)) = self.mouse_drag {
                    // Convert to normalized coordinates (-1 to 1)
                    let prev = Vector2::new(
                        (prev_x as f32 / width as f32) * 2.0 - 1.0,
                        (prev_y as f32 / height as f32) * 2.0 - 1.0,
                    );
                    let curr = Vector2::new(
                        (event.column as f32 / width as f32) * 2.0 - 1.0,
                        (event.row as f32 / height as f32) * 2.0 - 1.0,
                    );
                    self.camera.trackball_rotate(prev, curr);
                    self.mouse_drag = Some((event.column, event.row));
                }
            }
            MouseEventKind::Up(MouseButton::Left) => {
                self.mouse_drag = None;
            }
            MouseEventKind::ScrollUp => {
                self.camera.zoom_by(1.1);
            }
            MouseEventKind::ScrollDown => {
                self.camera.zoom_by(0.9);
            }
            _ => {}
        }
    }

    fn is_atom_visible(&self, atom: &crate::molecule::Atom) -> bool {
        // Check chain visibility
        let chain_idx = if atom.chain_id.is_ascii_uppercase() {
            (atom.chain_id as u8 - b'A') as usize
        } else if atom.chain_id.is_ascii_lowercase() {
            (atom.chain_id as u8 - b'a') as usize
        } else {
            0
        };
        if chain_idx < 26 && !self.chain_visible[chain_idx] {
            return false;
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

    fn render_molecule(&self, buffer: &mut PixelBuffer, pixels_per_cell: (usize, usize)) {
        let pixel_width = buffer.width() as f32;
        let pixel_height = buffer.height() as f32;
        let center_x = pixel_width / 2.0;
        let center_y = pixel_height / 2.0;

        let cell_width = pixel_width / pixels_per_cell.0 as f32;
        let cell_height = pixel_height / pixels_per_cell.1 as f32;
        let base_scale = cell_width.min(cell_height) * 0.4 * self.camera.zoom;
        let scale_x = base_scale * pixels_per_cell.0 as f32;
        let scale_y = base_scale * pixels_per_cell.1 as f32;
        let scale = (scale_x + scale_y) * 0.5;

        // Collect visible atoms with their screen positions per assembly instance
        let mut projected_instances: Vec<Vec<(usize, f32, f32, f32, (u8, u8, u8))>> = Vec::new();
        let mut z_min = f32::INFINITY;
        let mut z_max = f32::NEG_INFINITY;

        for instance in &self.current_assembly().instances {
            let mut projected: Vec<(usize, f32, f32, f32, (u8, u8, u8))> = Vec::new();
            for (idx, atom) in self.molecule.atoms.iter().enumerate() {
                if !self.is_atom_visible(atom) {
                    continue;
                }
                if !instance.applies_to_chain(atom.chain_id) {
                    continue;
                }

                let world = instance.transform.apply(atom.coord);
                let (screen_pos, z) = self.camera.project(world);
                let sx = center_x + screen_pos.x * scale_x;
                let sy = center_y + screen_pos.y * scale_y;

                z_min = z_min.min(z);
                z_max = z_max.max(z);

                // Determine color based on scheme
                let color = self.get_atom_color(atom, idx);

                projected.push((idx, sx, sy, z, color));
            }

            // Sort by depth (back to front)
            projected.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
            projected_instances.push(projected);
        }

        for projected in &projected_instances {
            // Render based on representation
            match self.representation {
                Representation::Backbone => {
                    self.render_backbone(buffer, projected, scale, z_min, z_max);
                }
                Representation::BallAndStick => {
                    self.render_ball_and_stick(buffer, projected, scale, z_min, z_max);
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
                for ss in &self.molecule.secondary_structure {
                    if ss.contains(atom.chain_id, atom.residue_seq) {
                        return match ss.ss_type {
                            SecondaryStructure::Helix(_) => (255, 0, 255), // Magenta
                            SecondaryStructure::Sheet => (255, 255, 0),    // Yellow
                            SecondaryStructure::Coil => (255, 255, 255),   // White
                        };
                    }
                }
                (255, 255, 255) // Default white for coil
            }
        }
    }

    fn render_backbone(
        &self,
        buffer: &mut PixelBuffer,
        projected: &[(usize, f32, f32, f32, (u8, u8, u8))],
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

        // Create a map from atom index to projected position
        let proj_map: std::collections::HashMap<usize, (f32, f32, f32, (u8, u8, u8))> = projected
            .iter()
            .map(|(idx, x, y, z, c)| (*idx, (*x, *y, *z, *c)))
            .collect();

        // Draw lines between consecutive backbone atoms in same chain
        let mut prev: Option<(char, i32, f32, f32, f32, (u8, u8, u8))> = None;

        for &idx in &backbone_indices {
            let atom = &self.molecule.atoms[idx];
            if let Some((x, y, z, color)) = proj_map.get(&idx) {
                let color = if self.shading_enabled {
                    depth_cue(*color, *z, z_max, z_min)
                } else {
                    *color
                };

                if let Some((prev_chain, prev_seq, px, py, pz, _pcolor)) = prev {
                    // Only connect if same chain and consecutive residue
                    if atom.chain_id == prev_chain && atom.residue_seq == prev_seq + 1 {
                        buffer.draw_line(px, py, pz, *x, *y, *z, color);
                    }
                }

                prev = Some((atom.chain_id, atom.residue_seq, *x, *y, *z, color));
            }
        }
    }

    fn render_ball_and_stick(
        &self,
        buffer: &mut PixelBuffer,
        projected: &[(usize, f32, f32, f32, (u8, u8, u8))],
        scale: f32,
        z_min: f32,
        z_max: f32,
    ) {
        use crate::render::braille::depth_cue;

        // Create a map from atom index to projected position
        let proj_map: std::collections::HashMap<usize, (f32, f32, f32, (u8, u8, u8))> = projected
            .iter()
            .map(|(idx, x, y, z, c)| (*idx, (*x, *y, *z, *c)))
            .collect();

        // Draw bonds first (behind atoms)
        for bond in &self.molecule.bonds {
            if let (Some(&(x1, y1, z1, c1)), Some(&(x2, y2, z2, _c2))) =
                (proj_map.get(&bond.atom1), proj_map.get(&bond.atom2))
            {
                let avg_z = (z1 + z2) / 2.0;
                let color = if self.shading_enabled {
                    depth_cue(c1, avg_z, z_max, z_min)
                } else {
                    c1
                };
                buffer.draw_line(x1, y1, z1, x2, y2, z2, color);
            }
        }

        // Draw atoms
        for (idx, x, y, z, color) in projected {
            let atom = &self.molecule.atoms[*idx];
            let radius = atom.vdw_radius() * scale * 0.15; // Scale down for visibility
            let color = if self.shading_enabled {
                depth_cue(*color, *z, z_max, z_min)
            } else {
                *color
            };
            buffer.draw_circle(*x, *y, *z, radius.max(1.0), color);
        }
    }

    fn render_surface(
        &self,
        buffer: &mut PixelBuffer,
        projected: &[(usize, f32, f32, f32, (u8, u8, u8))],
        scale: f32,
        z_min: f32,
        z_max: f32,
    ) {
        use crate::render::braille::depth_cue;

        let probe_radius = 1.4_f32;
        let surface_scale = 0.15_f32;

        for (idx, x, y, z, color) in projected {
            let atom = &self.molecule.atoms[*idx];
            let radius = (atom.vdw_radius() + probe_radius) * scale * surface_scale;
            let color = if self.shading_enabled {
                depth_cue(*color, *z, z_max, z_min)
            } else {
                *color
            };
            buffer.draw_circle(*x, *y, *z, radius.max(1.0), color);
        }
    }

    /// Render cartoon representation with secondary structure elements
    fn render_cartoon(
        &self,
        buffer: &mut PixelBuffer,
        projected: &[(usize, f32, f32, f32, (u8, u8, u8))],
        scale: f32,
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

        // Create a map from atom index to projected position
        let proj_map: std::collections::HashMap<usize, (f32, f32, f32, (u8, u8, u8))> = projected
            .iter()
            .map(|(idx, x, y, z, c)| (*idx, (*x, *y, *z, *c)))
            .collect();

        let helix_width = (2.6 * scale).max(2.0);
        let sheet_width = (2.2 * scale).max(1.8);
        let coil_width = (1.2 * scale).max(1.0);
        let arrow_length = (sheet_width * 1.6).max(3.0);
        let arrow_width = (sheet_width * 1.8).max(3.0);

        // Track previous residue for drawing
        let mut prev: Option<(char, i32, f32, f32, f32, (u8, u8, u8), SecondaryStructure)> = None;
        let mut last_segment_dir: Option<Vector2<f32>> = None;

        for &idx in &backbone_indices {
            let atom = &self.molecule.atoms[idx];
            if let Some((x, y, z, color)) = proj_map.get(&idx) {
                let ss = self.get_secondary_structure(atom.chain_id, atom.residue_seq);
                let color = if self.shading_enabled {
                    depth_cue(*color, *z, z_max, z_min)
                } else {
                    *color
                };

                if let Some((prev_chain, prev_seq, px, py, pz, prev_color, prev_ss)) = prev {
                    // Only connect if same chain and consecutive residue
                    if atom.chain_id == prev_chain && atom.residue_seq == prev_seq + 1 {
                        let segment_dir = Vector2::new(*x - px, *y - py);
                        last_segment_dir = Some(segment_dir);

                        // Determine ribbon width based on secondary structure
                        match (&prev_ss, &ss) {
                            (SecondaryStructure::Helix(_), _) | (_, SecondaryStructure::Helix(_)) => {
                                // Helices get thick ribbons
                                self.draw_ribbon(buffer, px, py, pz, *x, *y, *z, color, helix_width);
                            }
                            (SecondaryStructure::Sheet, _) | (_, SecondaryStructure::Sheet) => {
                                // Sheets get medium-width flat ribbons
                                self.draw_ribbon(buffer, px, py, pz, *x, *y, *z, color, sheet_width);
                            }
                            _ => {
                                // Coils get thin tubes
                                self.draw_ribbon(buffer, px, py, pz, *x, *y, *z, color, coil_width);
                            }
                        };

                        if matches!(prev_ss, SecondaryStructure::Sheet)
                            && !matches!(ss, SecondaryStructure::Sheet)
                        {
                            self.draw_arrowhead(
                                buffer,
                                px,
                                py,
                                pz,
                                segment_dir,
                                prev_color,
                                arrow_length,
                                arrow_width,
                            );
                        }
                    } else {
                        last_segment_dir = None;
                    }
                }

                prev = Some((atom.chain_id, atom.residue_seq, *x, *y, *z, color, ss));
            }
        }

        if let (Some((_, _, px, py, pz, color, ss)), Some(dir)) = (prev, last_segment_dir) {
            if matches!(ss, SecondaryStructure::Sheet) {
                self.draw_arrowhead(
                    buffer,
                    px,
                    py,
                    pz,
                    dir,
                    color,
                    arrow_length,
                    arrow_width,
                );
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
}

/// Run the interactive viewer
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

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(path, molecule);
    let backend = detect_backend();

    // Main loop
    let result = run_loop(&mut terminal, &mut app, backend);

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
    backend: RenderBackend,
) -> Result<(), UiError> {
    loop {
        // Calculate FPS
        let now = Instant::now();
        let delta = now.duration_since(app.last_frame);
        app.fps = 1.0 / delta.as_secs_f32();
        app.last_frame = now;

        // Auto-spin
        if app.auto_spin {
            let rotation_delta = Vector2::new(0.01, 0.0);
            app.camera.trackball_rotate(Vector2::zeros(), rotation_delta);
        }

        let size = terminal.size()?;
        let mut mol_height = if app.show_hud {
            size.height.saturating_sub(3)
        } else {
            size.height
        };
        if mol_height == 0 {
            mol_height = 1;
        }
        let mol_width = size.width.max(1);

        let image_active = matches!(backend, RenderBackend::Image(_)) && !app.show_help;
        let pixels_per_cell = if app.show_help {
            (1, 2)
        } else {
            match backend {
                RenderBackend::HalfBlock => (1, 2),
                RenderBackend::Image(output::ImageProtocol::Sixel) => (1, 6),
                RenderBackend::Image(_) => (2, 4),
            }
        };
        let pixel_width = mol_width as usize * pixels_per_cell.0;
        let pixel_height = mol_height as usize * pixels_per_cell.1;
        let mut buffer = PixelBuffer::new(pixel_width, pixel_height);
        buffer.clear();
        app.render_molecule(&mut buffer, pixels_per_cell);

        // Render
        terminal.draw(|f| {
            let buffer_ref = if image_active { None } else { Some(&buffer) };
            draw_ui(f, app, backend, mol_height, buffer_ref);
        })?;

        if image_active {
            if let RenderBackend::Image(protocol) = backend {
                render_image(protocol, &buffer, mol_width, mol_height, terminal.backend_mut())?;
            }
        }

        // Handle events with timeout for animation
        let timeout = if app.auto_spin {
            Duration::from_millis(16) // ~60 FPS
        } else {
            Duration::from_millis(100)
        };

        if event::poll(timeout)? {
            match event::read()? {
                Event::Key(key) => {
                    app.handle_key(key.code, key.modifiers);
                }
                Event::Mouse(mouse) => {
                    let size = terminal.size()?;
                    app.handle_mouse(mouse, size.width, size.height);
                }
                Event::Resize(_, _) => {
                    // Terminal resized, will re-render on next loop
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
                " {} | {} chains, {} atoms | {} | {} | Backend: {} | Asm: {} ({}/{}) | AltLoc: {} | FPS: {:.0} | q: quit, Tab: repr, c: color, /: asm, ': altloc, h: help, F1: HUD",
                app.filename,
                app.molecule.chain_count(),
                app.molecule.atom_count(),
                app.representation.name(),
                app.color_scheme.name(),
                backend.label(),
                app.current_assembly_id(),
                app.assembly_index + 1,
                app.molecule.assembly_count(),
                app.alt_loc_mode.label(),
                app.fps,
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
    "  Mouse left-drag : rotate",
    "  Scroll         : zoom",
    "  Arrow keys     : pan",
    "  Shift+Up/Down  : zoom in/out",
    "  0              : reset view",
    "  p              : toggle auto-spin",
    "",
    "Display",
    "  Tab            : cycle representation",
    "  c              : cycle color scheme",
    "  s              : toggle shading",
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
