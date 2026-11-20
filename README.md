# Monte Carlo Simulation of 2D Bearing-Only Object Localization

**Author:** Oleksandr Terentiev  
**Repository:** https://github.com/oterentiev/bearing-only-localization-monte-carlo  
**Version:** 1.0.0  
**License:** MIT

## Overview

This repository contains a Python implementation of a Monte Carlo computational experiment for evaluating two-dimensional bearing-only object localization. It compares a weighted nonlinear least-squares estimator (WLS) with an unweighted ordinary least-squares estimator (OLS) under observation configurations involving three and four bearing measurement stations.

The implementation uses a nonlinear bearing model, an analytical Jacobian, wrapped angular residuals, and iterative position refinement with a backtracking step-size procedure.

## Methodology

The target position is represented by the vector

$$
\mathbf{p} = [x, y]^T
$$

For a station located at horizontal coordinate \(s_i\) on the x-axis, the bearing model implemented in the code is

$$
\theta_i(\mathbf{p}) = \operatorname{atan2}(y, x-s_i)
$$

The measurement covariance matrix is diagonal and is constructed from the specified bearing standard deviations. WLS uses the inverse measurement covariance as its weight matrix. OLS uses the identity matrix.

For each noisy observation sample, the program iteratively estimates the target coordinates. It then evaluates estimation errors and covariance-based uncertainty measures.

## Experimental configurations

The program runs 10,000 Monte Carlo trials for each of the following configurations:

- **Three stations:** x-coordinates 0.0, 152.5, and 305.0 m; bearing standard deviations 0.10°, 0.10°, and 0.20°.
- **Four stations:** x-coordinates 0.0, 152.5, 305.0, and 457.5 m; bearing standard deviations 0.10°, 0.10°, 0.20°, and 0.10°.

The target position used in the simulation is (367.0, 214.0) m. The random-number generator is initialized with seed 20261120 to support reproducibility.

## Evaluation metrics

For both WLS and OLS, the program calculates:

- coordinate-wise bias;
- coordinate-wise RMSE and two-dimensional position RMSE;
- empirical standard deviations of coordinate errors;
- mean predicted standard deviations;
- 95% confidence-region coverage based on the normalized estimation error squared (NEES) and the chi-square threshold with two degrees of freedom.

## Requirements

- Python 3.9 or later
- NumPy
- SciPy
- Matplotlib

Install dependencies using:

```bash
python -m pip install -r requirements.txt
```

## Running the experiment

Run the program from a terminal:

```bash
python kalman_bearing_experiment.py
```

The program writes its output files to the same directory as the Python script.

## Output files

After a successful run, the program generates:

- `monte_carlo_summary.csv` — summary metrics for the three- and four-station configurations and both estimators;
- `bearing_geometry.png` — geometry of the bearing measurement stations and target;
- `uncertainty_3_stations.png` — Monte Carlo localization errors and the theoretical 95% covariance ellipse for the three-station configuration;
- `uncertainty_4_stations.png` — corresponding plot for the four-station configuration.

The program also prints noiseless bearing angles, summary metrics, and a deterministic reference-example estimate to the console.

## Reproducibility

The random seed and experiment parameters are defined in the source code. For exact reproduction, use the same software version and dependency environment.

## Citation

If you use this software in academic work, please cite the archived release in Zenodo using the DOI assigned to that release. The DOI will be added here after the first Zenodo publication.

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
