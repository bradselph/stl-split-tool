# STL Split Tool

STL mesh splitting utility for 3D printing preparation. Automatically detects symmetry axes or allows manual plane selection to split 3D models into two watertight halves suitable for printing.

## Features

- Automatic symmetry axis detection
- Manual split plane configuration with live preview
- Interactive reselection after preview
- Robust mesh sealing with multiple repair strategies
- Watertight mesh validation
- Support for both capped and uncapped splits
- Visual plane preview before splitting

## Requirements

- Python 3.7 or higher
- See requirements.txt for package dependencies

## Installation

```bash
git clone https://github.com/bradselph/stl-split-tool
cd stl-split-tool
pip install -r requirements.txt
```

## Usage

### Interactive Mode

Run without arguments to enter interactive mode:

```bash
python split_mirror.py
```

The tool will guide you through:
1. File selection
2. Output directory configuration
3. Split mode selection (Auto or Manual)
4. Preview and confirmation
5. Processing and export

### Command Line Mode

Specify input file and optional output directory:

```bash
python split_mirror.py input.stl
python split_mirror.py input.stl output_directory
```

### Manual Mode

Manual mode provides:
- Visual preview of split plane
- Four positioning methods:
  1. Centroid
  2. Bounding box center
  3. Custom coordinate
  4. Percentage-based position
- Reselection option if preview is incorrect
- Confirmation prompt with y/n/q options

## How It Works

### Automatic Detection

The tool analyzes vertex distribution across X, Y, and Z axes to identify the most likely symmetry axis. A confidence score indicates detection reliability.

### Mesh Sealing

Multiple sealing strategies ensure watertight output:
- Hole filling via trimesh repair
- Normal fixing and inversion correction
- Degenerate face removal
- Duplicate face removal
- Vertex merging
- Manual boundary edge detection with planar triangulation

### Split Process

1. Validates input mesh
2. Determines split plane (auto or manual)
3. Slices mesh along plane
4. Caps open edges with triangulated surfaces
5. Applies sealing strategies
6. Validates watertight status
7. Exports both halves as STL files

## Output

Generated files follow naming convention:
- `original_name_half_A.stl`
- `original_name_half_B.stl`

Both files are checked for watertight status. Non-watertight meshes receive repair recommendations.

## Troubleshooting

### "No available triangulation engine" Error

Install the triangulation dependency:

```bash
pip install mapbox-earcut
```

### Non-Watertight Output

If meshes are not watertight after processing:
1. Import into mesh repair software (Meshmixer, Netfabb, or Microsoft 3D Builder)
2. Use automatic repair functions
3. Verify watertight status before printing

### Low Confidence Warning

Automatic detection may struggle with:
- Asymmetric models
- Complex geometries
- Irregular vertex distribution

Use manual mode for better control.

## Dependencies

- trimesh: Core mesh manipulation
- numpy: Numerical operations
- matplotlib: Visualization
- scipy: Delaunay triangulation
- mapbox-earcut: Polygon triangulation for capping