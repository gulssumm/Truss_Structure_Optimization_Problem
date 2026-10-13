# Truss Structure Weight Optimization
This repository benchmarks classic metaheuristic algorithms against a surrogate-assisted evolutionary approach for minimizing the total structural weight of a truss system subject to mechanical stress and nodal displacement constraints.

## Problem Formulation
The objective is to find the optimal cross-sectional areas $X = [A_1, A_2, \dots, A_n]$ that minimize the penalized structural weight function:

$$F(X) = \rho \sum_{i=1}^{n} A_i L_i + P \cdot \left( C_{\text{stress}}(X) + C_{\text{disp}}(X) \right)$$

where $\rho$ is material density, $L_i$ is member length, and $P$ is the penalty factor applied whenever member stresses or nodal displacements exceed allowable engineering thresholds.

## Implemented Optimization Methods

| File | Method | Key Implementation Details |
| :--- | :--- | :--- |
| `Task1.py` | **Genetic Algorithm (GA)** | Population initialization, tournament selection, single-point crossover, random mutation, and elitist preservation. |
| `Task2.py` | **Particle Swarm Optimization (PSO)** | Swarm velocity and position updates guided by personal best (`pbest`) and global best (`gbest`) structural configurations. |
| `Task2_SA.py` | **Simulated Annealing (SA)** | Single-solution trajectory search using geometric temperature cooling schedules and Boltzmann acceptance criteria to escape local minima. |
| `Task3.py` | **Kriging Metamodel-Assisted GA** | Trains a **Gaussian Process Regressor** (`RBF` + `ConstantKernel` via `scikit-learn`) on `truss_dataset.csv` (`Task3_DataGenerator.py`) to approximate computationally expensive structural evaluations during GA evolution. |

## Tech Stack
- **Language:** Python 3.x
- **Libraries:** `scikit-learn` (`GaussianProcessRegressor`), `numpy`, `pandas`

## Usage
```bash
python Task1.py       # Run Baseline Genetic Algorithm
python Task2.py       # Run Particle Swarm Optimization
python Task2_SA.py    # Run Simulated Annealing
python Task3.py       # Run Kriging Surrogate-Assisted GA
