# Shock-Wave Structure Solver

This repository contains a Python-based solver for **shock-wave structures in real fluids**, with a focus on carbon dioxide (CO₂). It uses NIST thermodynamic tables for accurate fluid properties and computes the downstream states and shock thickness for given upstream conditions.

---

## Features

* Loads CO₂ data from NIST `.txt` files (`data/` folder).
* Calculates upstream and downstream properties using the **Rankine-Hugoniot relations**.
* Uses **boundary value problem (BVP) solvers** to compute smooth shock profiles.
* Supports plotting and saving results (velocity, temperature, pressure, density) in both **PNG** and **EPS** formats.
* Saves processed data in **CSV** files for further analysis.

---

## Requirements

* Python 3.8+
* Packages:

  * `numpy`
  * `pandas`
  * `matplotlib`
  * `scipy`

Install required packages with:

```
pip install numpy pandas matplotlib scipy
```

---

## Folder Structure

```
Shock_Wave_Structure_Solver/
├── data/         # NIST .txt tables for CO₂
├── figures/      # Automatically generated plots and CSV data
├── solver.py     # Main Python solver script
└── README.md     # This file
```

> The `data/` folder must contain the CO₂ NIST data files starting with `CO2_` and ending with `.txt`.

---

## Usage

1. Place all NIST `.txt` files in the `data/` folder.
2. Run the main Python script:

```
python solver.py
```

3. Generated plots and CSV data are saved automatically in the `results/` folder.

---

### Example Outputs

* Velocity profile (Mach number) across the shock.
* Temperature normalized by critical temperature.
* Pressure normalized by critical pressure.
* Density normalized by critical density.
* Combined figure showing all variables and calculated shock thickness.

---

## Customization

* **`DATA_DIR`**: Folder containing NIST `.txt` files (default: `data/`).
* **`SAVE_DIR`**: Folder where figures and CSVs are saved (default: `figures/`).
* **Upstream conditions** (in the script): Mach number, temperature, and pressure can be modified.
* **Solver parameters**: Number of points, tolerance, and amplitude for initial guess can be adjusted for accuracy/performance.

---

## Citation

If you use this code in your research, please cite the repository:

Foley-Valledor, K. (2026). *Shock-Wave Structure Solver*. GitHub. [https://github.com/foleyvak/Shock_Wave_Structure_Solver](https://github.com/foleyvak/Shock_Wave_Structure_Solver)

---

## Notes

* Designed to be **self-contained**; no modification to the local environment is required beyond placing NIST data in the `data/` folder.
* Handles errors in NIST data interpolation gracefully.
* Computes shock thickness using the maximum density gradient.
