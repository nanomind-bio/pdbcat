# pdbcat

```
pdbcat 1ubq.pdb
```

![pdbcat output](assets/iterm2-cartoon.png)

A quick terminal viewer for PDB and mmCIF files. **Not a replacement for PyMOL or ChimeraX**—just a fast way to glance at a structure without leaving the terminal.

## Examples

### iTerm2 Backend (high resolution inline images)

| Cartoon | Ball-and-Stick | Surface | Backbone |
|---------|----------------|---------|----------|
| ![cartoon](assets/iterm2-cartoon.png) | ![bas](assets/iterm2-bas.png) | ![surface](assets/iterm2-surface.png) | ![backbone](assets/iterm2-backbone.png) |

### Half-Block Backend (works in any terminal)

| Cartoon | Ball-and-Stick | Surface | Backbone |
|---------|----------------|---------|----------|
| ![cartoon](assets/half-cartoon.png) | ![bas](assets/half-bas.png) | ![surface](assets/half-surface.png) | ![backbone](assets/half-backbone.png) |

### Color Schemes

| Chain | Rainbow | Secondary Structure |
|-------|---------|---------------------|
| ![chain](assets/color-chain.png) | ![rainbow](assets/color-rainbow.png) | ![secondary](assets/color-secondary.png) |

## Installation

```bash
cargo install --path .
```

Or build from source:

```bash
cargo build --release
./target/release/pdbcat structure.pdb
```

## Usage

```bash
# Quick view (auto-detects terminal, renders to stdout)
pdbcat structure.pdb

# Different representations
pdbcat structure.pdb --repr cartoon      # default
pdbcat structure.pdb --repr ball-and-stick
pdbcat structure.pdb --repr surface
pdbcat structure.pdb --repr backbone

# Color schemes
pdbcat structure.pdb -c chain            # default
pdbcat structure.pdb -c rainbow
pdbcat structure.pdb -c secondary

# Background options
pdbcat structure.pdb --bg black          # default
pdbcat structure.pdb --bg white
pdbcat structure.pdb --bg transparent

# Force specific backend
pdbcat structure.pdb --backend iterm2
pdbcat structure.pdb --backend half

# Interactive mode (rotate with mouse, keyboard controls)
pdbcat structure.pdb -i

# Export to PNG
pdbcat structure.pdb -o output.png
pdbcat structure.pdb -o output.png -r 1920x1080
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `FILE` | Path to PDB or mmCIF file |
| `-i, --interactive` | Interactive mode with keyboard/mouse controls |
| `-o, --output PATH` | Export to PNG file |
| `-r, --resolution WxH` | Resolution (default: terminal size for stdout, 800x600 for -o) |
| `--repr MODE` | Representation: cartoon, ball-and-stick, surface, backbone |
| `-c, --color SCHEME` | Color scheme: chain, rainbow, secondary |
| `--no-shading` | Disable shading (flat colors) |
| `--bg COLOR` | Background: white, black (default), transparent |
| `--backend MODE` | Force render backend: iterm2, half (auto-detect if not set) |

## Interactive Mode Controls

| Key | Action |
|-----|--------|
| Mouse drag | Rotate |
| Scroll / `[` `]` | Zoom |
| Arrow keys | Pan |
| Tab | Cycle representation |
| `c` | Cycle color scheme |
| `0` | Reset view |
| `q` / Esc | Quit |

## Supported Formats

- **PDB** (.pdb)
- **mmCIF** (.cif, .mmcif)

## Terminal Compatibility

| Terminal | Backend | Quality |
|----------|---------|---------|
| iTerm2 | Inline images | Best |
| Any terminal | Half-block characters | Good |

The backend is auto-detected. Use `--backend` to override.

## License

MIT
