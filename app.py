import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io

# Import your backend modules
from spm_atombridge.adapters.graph_adapter import GraphAdapter
from spm_atombridge.adapters.ase_adapter import ASEAdapter
from spm_atombridge.core.comparison import compare_images
from spm_atombridge.adapters.plotting_adapter import PlottingAdapter
from spm_atombridge.core.image import Image

# Set page config
st.set_page_config(page_title="SPM-AtomBridge", page_icon="⚛️", layout="wide")

st.title("⚛️ SPM-AtomBridge")
st.markdown("""
**A unified environment for defect-to-spectroscopy analysis across DFT, LAMMPS, and Scanning Probe Microscopy.**

Upload your data below to generate publication-ready graphs and comparisons.
""")

# Sidebar for navigation
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox("Choose a module", [
    "Home",
    "1. Structure & Defects",
    "2. Graph LAMMPS Log",
    "3. Compare Images"
])

if app_mode == "Home":
    st.header("Welcome to SPM-AtomBridge")
    st.write("Select a module from the sidebar to get started.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/AFM_schematic.jpg/800px-AFM_schematic.jpg", caption="Atomic Force Microscopy", width="stretch")

elif app_mode == "1. Structure & Defects":
    st.header("Structure & Defect Engineering")
    st.write("Build a surface and create a vacancy using ASE.")
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Element", ["Au", "Cu", "Ag", "Pt"])
        size = st.slider("Slab size (x, y, layers)", 2, 6, 3)
        vacuum = st.slider("Vacuum (Å)", 5.0, 20.0, 10.0)
        
    if st.button("Build Surface"):
        adapter = ASEAdapter()
        struct = adapter.build_fcc111(symbol, size=(size, size, 3), vacuum=vacuum)
        st.success(f"Built {struct.summary()['formula']} with {struct.summary()['n_atoms']} atoms.")
        
        # Save to XYZ and offer download
        out_path = Path("spm_atombridge/examples/_output/web_build.xyz")
        out_path.parent.mkdir(exist_ok=True, parents=True)
        struct.save(out_path)
        
        with open(out_path, "r") as f:
            st.download_button("Download Structure (.xyz)", f.read(), file_name="structure.xyz")

elif app_mode == "2. Graph LAMMPS Log":
    st.header("LAMMPS Thermodynamics")
    st.write("Upload a LAMMPS `log.lammps` file to plot Temperature and Energy.")
    
    uploaded_file = st.file_uploader("Choose a LAMMPS log file", type=["lammps", "log", "txt"])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = Path("spm_atombridge/examples/_output/temp_log.lammps")
        temp_path.parent.mkdir(exist_ok=True, parents=True)
        temp_path.write_bytes(uploaded_file.getvalue())
        
        try:
            ga = GraphAdapter()
            out_png = Path("spm_atombridge/examples/_output/web_thermo.png")
            ga.plot_lammps_thermo(temp_path, out_png)
            st.image(str(out_png), caption="LAMMPS Thermo Plot", width="stretch")
        except Exception as e:
            st.error(f"Error parsing file: {e}")

elif app_mode == "3. Compare Images":
    st.header("Compare Images (e.g., Simulation vs Experiment)")
    st.write("Upload two images (as .npy arrays) to compute correlation and flag discrepancies.")
    
    col1, col2 = st.columns(2)
    with col1:
        file_a = st.file_uploader("Upload Image A (Simulation)", type=["npy"])
    with col2:
        file_b = st.file_uploader("Upload Image B (Experiment)", type=["npy"])
        
    if file_a and file_b:
        # Load arrays
        img_a = np.load(file_a)
        img_b = np.load(file_b)
        
        if img_a.shape != img_b.shape:
            st.error(f"Shape mismatch: {img_a.shape} vs {img_b.shape}")
        else:
            res = compare_images(img_a, img_b)
            
            st.metric("Correlation", f"{res.correlation:.4f}")
            st.metric("RMSD", f"{res.rmsd:.4f}")
            
            # Plot side by side
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
            axes[0].imshow(img_a, cmap="afmhot"); axes[0].set_title("Image A")
            axes[1].imshow(img_b, cmap="afmhot"); axes[1].set_title("Image B")
            axes[2].imshow(res.residual_map, cmap="coolwarm"); axes[2].set_title("Residual")
            for ax in axes: ax.axis("off")
            st.pyplot(fig)
            
            # Download report
            report = f"Correlation: {res.correlation:.4f}\nRMSD: {res.rmsd:.4f}\nFlagged Pixels: {res.discrepancy_mask.sum()}"
            st.download_button("Download Report", report, file_name="comparison_report.txt")
