# pdbcat - Terminal PDB/CIF Molecular Structure Viewer

**Version:** 1.0
**License:** MIT

## Overview

`pdbcat` is a fast, keyboard-driven terminal-based viewer for PDB and mmCIF molecular structure files. It renders protein and nucleic acid structures using a pixel buffer with optional image-protocol output (kitty/iTerm2/sixel) and a half-block fallback.

```
pdbcat protein.pdb
pdbcat structure.cif
```

## Core Design Principles

- **Fast** - Renders interactively in the console
- **Keyboard-centric** - Single-key mnemonics, no modal interface
- **High detail color** - Image protocols when supported, half-block fallback otherwise
- **Visual parity** - Renders similarly to 3Dmol.js (no API compatibility required)

---

## Technical Specifications

### Implementation Stack

- **Language:** Rust
- **Terminal library:** crossterm + ratatui
- **Threading:** Full parallel pipeline (parallel transforms + tiled rendering + async I/O)

### Rendering Engine

#### Projection
- **Model:** Orthographic (no perspective distortion)
- **Rotation:** Trackball (arcball) - intuitive sphere-based rotation where rotation depends on click position

#### Pseudo-Pixel System
- **Primary method:** Image protocols (Kitty Graphics, iTerm2 inline images, Sixel)
- **Fallback:** Half-block characters (1×2 subcells per terminal cell)
- **Internal raster:** Pixel buffer with per-backend scaling (typically 2×4 per cell for image protocols)
- **Depth cueing:** Brightness gradient (near atoms = bright, far atoms = dim)
- **Background:** Terminal default where possible; image backends use alpha where supported

#### Shading
- **Algorithm:** Ambient occlusion approximation (optional, toggle with `s`)
- **Default:** Off (performance consideration)

#### Progressive Rendering (for large structures)
- **Strategy:** LOD (Level of Detail) by distance from camera
- Near atoms rendered at full detail
- Far atoms simplified/subsampled
- No upper bound on structure size - progressive refinement on interaction stop

---

## Molecular Representations

### Supported Representations
1. **Backbone trace** - Cα atoms connected (proteins) / phosphate backbone (nucleic acids)
2. **Ball-and-stick** - Atoms as spheres with bond cylinders
3. **Cartoon/ribbon** - Secondary structure visualization (helix/sheet/coil)
4. **Surface** - Solvent Excluded Surface (SES/Connolly)

### Representation Cycling
- Press `Tab` to cycle through: backbone → ball-and-stick → cartoon → surface

### Atom Radii
- **Style:** Element-specific Van der Waals radii
- C: 1.70Å, N: 1.55Å, O: 1.52Å, S: 1.80Å, H: 1.20Å, etc.

### Bond Determination
- **Method:** Residue topology library
- Standard amino acid and nucleic acid bond templates
- Distance-based heuristic for unknown residues/ligands

### Bond Rendering
- **Style:** Depth-faded (bonds fade with depth like atoms)

### Backbone Gaps
- **Handling:** Visual break (no connection drawn across missing residues)

---

## Coloring

### Default Scheme
- **Multi-chain:** Distinct hue per chain (Chain A=red, B=blue, C=green, etc.)
- **Palette:** High-contrast categorical (bold primaries)

### Available Schemes
1. **Chain** - Distinct color per chain
2. **Rainbow** - N-terminus (blue) → C-terminus (red)
3. **Secondary structure** - Helix=magenta, Sheet=yellow, Coil=white

### Scheme Cycling
- Press `c` to cycle through color schemes

---

## Structure Handling

### File Formats
- **PDB** (.pdb)
- **mmCIF** (.cif, .mmcif)
- **Priority:** Exact filename given (no auto-detection between formats)

### Parsing
- **Strategy:** Strict parse, permissive render
- Exit with clear error on parse failure
- Tolerate missing atoms during rendering

### Secondary Structure Assignment
- **Primary:** Parse from file (HELIX/SHEET records in PDB, _struct_conf in mmCIF)
- **Fallback:** Calculate using DSSP algorithm

### Multi-Model Files (NMR)
- **Behavior:** Show first model only (MODEL 1)

### Biological Assemblies (mmCIF)
- **Support:** Can display all assemblies
- **Toggle:** `/` key cycles through assemblies

### Alternate Conformers (altloc)
- **Default:** Show A conformer
- **Toggle:** `'` (apostrophe) cycles through A/B/all

### Heteroatoms
- **Default:** Hidden
- **Ligands toggle:** `l` key
- **Waters toggle:** `F3` key

### Nucleic Acids (DNA/RNA)
- **Representation:** Nucleotide blocks - bases shown as filled shapes
- **Coloring:** A/T/G/C/U with distinct colors

---

## User Interface

### HUD (Heads-Up Display)
- **Style:** Customizable panels (configured in config file)
- **Toggle:** `F1`
- **Available panels:**
  - File name and path
  - Chain count and atom count
  - Current representation mode
  - Current color scheme
  - FPS counter
  - Memory usage
  - Keybinding hints

### Help System
- **Style:** Context-sensitive hints in bottom status bar

### Selection
- **Support:** None (view-only, no atom picking)

---

## Controls

### Mouse Controls
- **Mapping:** Configurable in config file
- **Default suggestion:**
  - Left-drag: Rotate (trackball)
  - Middle-drag: Pan
  - Right-drag or scroll: Zoom

### Keyboard Controls

#### View Manipulation
| Key | Action |
|-----|--------|
| Arrow keys | Pan (move structure in screen plane) |
| Shift+Up | Zoom in |
| Shift+Down | Zoom out |
| Mouse drag | Rotate (trackball) |
| `0` | Reset view (fit structure to terminal) |
| `p` | Toggle auto-spin |

#### Representation & Color
| Key | Action |
|-----|--------|
| `Tab` | Cycle representations (backbone→stick→cartoon→surface) |
| `c` | Cycle color schemes (chain→rainbow→secondary) |
| `s` | Toggle shading (ambient occlusion) |

#### Visibility Toggles
| Key | Action |
|-----|--------|
| `A`, `B`, `C`... | Toggle chain A, B, C visibility |
| `l` | Toggle ligands |
| `F3` | Toggle waters |
| `/` | Cycle assemblies |
| `'` | Cycle altloc conformers (A/B/all) |

#### Interface
| Key | Action |
|-----|--------|
| `F1` | Toggle HUD |
| `h` | Toggle help menu |
| `q` or `Esc` | Quit |

### Animation
- **Rotation response:** Instant (no interpolation/easing)
- **Auto-spin:** Toggle with `p`, continuous rotation until stopped

---

## Configuration

### File Format
- **Format:** JSON with comments (JSONC)
- **Location:** `~/.config/pdbcat/config.jsonc` (XDG-compliant)

### Configurable Options
```jsonc
{
  // Mouse button mapping
  "mouse": {
    "left_drag": "rotate",
    "middle_drag": "pan",
    "right_drag": "zoom",
    "scroll": "zoom"
  },

  // Default color scheme: "chain", "rainbow", "secondary"
  "default_color_scheme": "chain",

  // Default representation: "backbone", "stick", "cartoon", "surface"
  "default_representation": "cartoon",

  // HUD panels to display
  "hud_panels": ["filename", "stats", "representation", "color", "hints"],

  // Custom chain color palette (high-contrast categorical)
  "chain_colors": ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#FF00FF", "#00FFFF"],

  // Auto-spin speed (degrees per frame)
  "auto_spin_speed": 1.0,

  // Shading enabled by default
  "shading_default": false
}
```

### Terminal Capability
- **Detection:** Auto-detect image protocol support and color capability on startup
- **Override:** Environment variable `PDBCAT_COLOR=256|true|none`
- **Override:** Environment variable `PDBCAT_IMAGE=kitty|iterm2|sixel|half-block|none`
- **tmux note:** For image protocols inside tmux, enable passthrough and export iTerm2 variables in tmux `update-environment`
- **Runtime toggle:** Available via keybinding to cycle color modes

---

## Performance Specifications

### Target Performance
- **Goal:** Unlimited structure size via progressive rendering
- **Strategy:** LOD by distance from camera
- **Threading:** Full parallel pipeline
  - Parallel 3D→2D coordinate transforms
  - Tiled rendering
  - Async file I/O

### Memory
- **Limit:** None (load entire structure)
- **Large structures:** Rely on system memory, page to swap if needed

### Resize Behavior
- **Handling:** Debounced re-render (wait for resize to settle)

---

## Center of Rotation
- **Method:** Geometric center of visible atoms
- **Behavior:** Recalculates as visibility changes (e.g., toggling chains)

---

## CLI Interface

### Usage
```
pdbcat <file>
```

### Arguments
- `file` - Path to PDB or mmCIF file (required)

### Behavior
- **Missing file:** Exit with error code 1 and message to stderr
- **Invalid file:** Exit with parse error and code 1
- **Stdin:** Not supported (file path required)

### Initial State
- **Zoom:** Fit structure to terminal
- **Rotation:** Default orientation
- **Representation:** Per config (default: cartoon)
- **Color:** Per config (default: chain)

---

## Error Handling

### File Errors
| Condition | Behavior |
|-----------|----------|
| File not found | Exit code 1, error to stderr |
| Parse error | Exit code 1, descriptive error message |
| Invalid format | Exit code 1, specify expected formats |

### Runtime Errors
| Condition | Behavior |
|-----------|----------|
| Missing atoms | Skip silently, render what's available |
| Missing secondary structure | Fall back to DSSP calculation |
| Missing bonds | Calculate from topology + distance |

---

## Test Cases

### Example Files (`./example/`)

The following structures from RCSB PDB are provided for testing:

| File | PDB ID | Description | Chains | Resolution | Test Purpose |
|------|--------|-------------|--------|------------|--------------|
| `1UBQ.pdb`, `1UBQ.cif` | 1UBQ | Human ubiquitin | 1 (A) | 1.8Å | Single-chain protein, basic rendering |
| `7EOW.pdb`, `7EOW.cif` | 7EOW | Caplacizumab nanobody + vWF A1 domain | 2 (A, B) | 1.6Å | Multi-chain coloring, nanobody structure |
| `6ARU.pdb`, `6ARU.cif` | 6ARU | Cetuximab Fab + EGFR extracellular domain | 3+ (A, B, C, glycans) | 3.2Å | Complex multi-chain, heteroatoms, glycosylation |

### Feature Test Matrix

| Feature | 1UBQ | 7EOW | 6ARU |
|---------|------|------|------|
| PDB parsing | ✓ | ✓ | ✓ |
| mmCIF parsing | ✓ | ✓ | ✓ |
| Single chain | ✓ | | |
| Multi-chain coloring | | ✓ | ✓ |
| Chain toggle (A/B/C keys) | | ✓ | ✓ |
| Secondary structure (HELIX/SHEET) | ✓ | ✓ | ✓ |
| Backbone representation | ✓ | ✓ | ✓ |
| Ball-and-stick | ✓ | ✓ | ✓ |
| Cartoon/ribbon | ✓ | ✓ | ✓ |
| Surface rendering | ✓ | ✓ | ✓ |
| Ligand toggle (`l` key) | | | ✓ |
| Heteroatom handling | | | ✓ |
| Depth cueing | ✓ | ✓ | ✓ |

### Suggested Test Scenarios

1. **Basic Loading**
   ```
   pdbcat example/1UBQ.pdb
   pdbcat example/1UBQ.cif
   ```
   - Verify structure loads and renders
   - Check initial zoom fits structure to terminal

2. **Multi-Chain Visualization**
   ```
   pdbcat example/7EOW.pdb
   ```
   - Press `c` to cycle color schemes - verify chain colors differ
   - Press `A` and `B` to toggle individual chains
   - Verify center of rotation updates when hiding chains

3. **Complex Structure**
   ```
   pdbcat example/6ARU.pdb
   ```
   - Verify all 3 protein chains render with distinct colors
   - Press `l` to toggle ligands (glycans should appear/disappear)
   - Press `Tab` to cycle representations - verify performance with larger structure

4. **Format Parity**
   - Load same structure in PDB and mmCIF format
   - Visual output should be identical
   - Secondary structure should match

5. **Representation Cycling**
   - For each example file, press `Tab` to cycle through:
     backbone → ball-and-stick → cartoon → surface
   - Verify each representation renders correctly

---

## Future Considerations (Out of Scope for v1.0)

- Export functionality (PNG, ANSI, GIF)
- Selection and atom picking
- Stdin/pipe input
- URL fetching
- Trajectory/animation playback
- Clipping planes/slab mode
- 3Dmol.js API compatibility
