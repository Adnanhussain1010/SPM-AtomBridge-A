# SPM-AtomBridge

A unified environment for defect-to-spectroscopy analysis across DFT, LAMMPS, and Scanning Probe Microscopy (AFM/KPFM).

## Features
- Unified Data Ingestion: Read Gwyddion (.gwy), Bruker (.spm), ASE-readable structures, LAMMPS dumps, and DFT outputs.
- Defect Engineering: Build surfaces, create vacancies, and add adsorbates automatically.
- Charge Transfer & CPD: Wrap BaderKit and pymatgen to compute Bader charges and work-function-derived CPD maps.
- Simulation-Informed Imaging: Render theoretical AFM images via ppafm and compare them directly with experiment.
- Publication-Quality Graphs: Generate DOS, band structure, and LAMMPS thermodynamic plots automatically.
- Quantitative Comparison: Compute cross-correlation, RMSD, and automatically flag discrepancies between simulation and experiment.

## Installation
pip install spm-atombridge

For optional features:
pip install spm-atombridge[bader,ovito,kpfm]

## License
MIT License.
