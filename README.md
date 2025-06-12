# vdW_structures

## Overview
`vdW_structures` is a Python package built on `ase` and `pymatgen` that facilitates the construction of van der Waals (vdW) layered materials. It requires 'jupyter' and 'py3Dmol' for testing and visualization. 

## Features
- Allows the user to extract and recombine individual layers of vdW materials.
- Improves `pymatgen`'s CoherentInterfaceBuilder() handling of vdW heterostructure construction.
- Supports common vdW structural transformations (e.g., x, y and z-direction shifts of individual layers).

- NOTE: Solving for and setting vdW layers is currently only supported along c-lattice vectors with no x or y components, e.g., [0, 0, z]. 

## Installation
You can install `vdW_structures` using pip:

```bash
pip install git+https://github.com/yourusername/vdW_structures.git
```

Alternatively, if you'd like to make local changes to the package, clone the repository and install it in editable mode. 

```bash
git clone https://github.com/yourusername/vdW_structures.git
cd vdW_structures
pip install -e .
```

## License
This project is licensed under the MIT License. 

## Contact
For any questions or feedback, please reach out via GitHub Issues or email: rym@ornl.gov
