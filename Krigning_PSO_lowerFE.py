# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 23:20:38 2025

@author: ZEYNEP
"""

import random
import pandas as pd
import numpy as np
import time
from numpy.linalg import solve, LinAlgError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- GLOBAL CONSTANTS ---
E = 10.0E6   # Young's Modulus (psi)
P = 10**6    # Penalty term (used to create the 10^10 wall)
rho = 0.1    # Material density
NUM_DESIGN_VARIABLES = 10
DATASET_FILENAME = 'truss_dataset_SA.csv'

# Design Limits
A_MIN = 0.01
A_MAX = 35.0

# Truss Geometry and Boundary Conditions
NODES = np.array([[720, 360], [360, 360], [720, 0], [360, 0], [0, 360], [0, 0]])
CONNECTIVITY = np.array([[4, 5], [0, 1], [5, 3], [2, 3], [4, 1], [5, 0], [1, 3], [3, 2], [4, 3], [1, 2]])
L_BARS = np.array([360.0, 360.0, 360.0, 360.0, 509.1168, 720.0, 360.0, 360.0, 509.1168, 509.1168])
F_GLOBAL = np.zeros(12)
F_GLOBAL[3], F_GLOBAL[7] = -100000, -100000
CONSTRAINED_DOFS = np.array([8, 9, 11])

# Physical Constraint Limits
SIGMA_LIMIT = 25000.0 # psi
DELTA_LIMIT = 2.0     # inches

# --- TRUE FITNESS CALCULATION (STIFFNESS MATRIX) ---
def calculate_fitness(individual):
    """Calculates the True Fitness (Weight + Penalty) using the Stiffness Matrix method."""
    A = np.array(individual)
    num_nodes = 6
    num_bars = 10

    W_X = np.sum(A * L_BARS) * rho
    K_global = np.zeros((num_nodes * 2, num_nodes * 2))
    C, S = np.zeros(num_bars), np.zeros(num_bars)

    # 1. Build Global Stiffness Matrix
    for i in range(num_bars):
        n1_idx, n2_idx = CONNECTIVITY[i, 0], CONNECTIVITY[i, 1]
        dx = NODES[n2_idx, 0] - NODES[n1_idx, 0]
        dy = NODES[n2_idx, 1] - NODES[n1_idx, 1]
        L = L_BARS[i]
        C[i], S[i] = dx / L, dy / L
        T = np.array([[C[i], S[i], 0, 0], [0, 0, C[i], S[i]]])
        k_local = (E * A[i] / L) * np.array([[1, -1], [-1, 1]])
        k_global_element = T.T @ k_local @ T
        dof_indices = np.array([2*n1_idx, 2*n1_idx+1, 2*n2_idx, 2*n2_idx+1])

        for r_idx in range(4):
            for c_idx in range(4):
                K_global[dof_indices[r_idx], dof_indices[c_idx]] += k_global_element[r_idx, c_idx]

    # 2. Solve for Displacements (Q)
    all_dofs = np.arange(12)
    unconstrained_dofs = np.delete(all_dofs, CONSTRAINED_DOFS)
    K_reduced = K_global[np.ix_(unconstrained_dofs, unconstrained_dofs)]
    F_reduced = F_GLOBAL[unconstrained_dofs]

    try:
        Q_reduced = solve(K_reduced, F_reduced)
    except LinAlgError:
        return W_X + P * 1e4 # Instability Penalty (10^10 level)

    Q_full = np.zeros(12)
    Q_full[unconstrained_dofs] = Q_reduced

    # 3. Calculate Stresses (Sigma)
    Sigma_bars = np.zeros(num_bars)
    for i in range(num_bars):
        q_element = Q_full[[2*CONNECTIVITY[i, 0], 2*CONNECTIVITY[i, 0]+1, 2*CONNECTIVITY[i, 1], 2*CONNECTIVITY[i, 1]+1]]
        stress_matrix = np.array([-C[i], -S[i], C[i], S[i]])
        Sigma_bars[i] = (E / L_BARS[i]) * np.dot(stress_matrix, q_element)

    # 4. Numerical Instability Check
    max_abs_stress = np.max(np.abs(Sigma_bars))
    Q_unconstrained = np.delete(Q_full, CONSTRAINED_DOFS)
    max_abs_displacement = np.max(np.abs(Q_unconstrained))

    if max_abs_stress > 1e10 or max_abs_displacement > 1e6:
        return W_X + P * 1e4

    # 5. Constraint Violation Calculation (Penalty)
    C_stress_X = np.sum(np.maximum(0, (np.abs(Sigma_bars) / SIGMA_LIMIT) - 1))
    critical_displacements = Q_full[[3, 7]]
    C_disp_X = np.sum(np.maximum(0, (np.abs(critical_displacements) / DELTA_LIMIT) - 1))

    Penalty_Term = P * (C_stress_X + C_disp_X)

    # 6. Final Fitness
    return W_X + Penalty_Term

# --- KRIGING MODEL TOOLS ---
def kriging_predict_value(model, individual, scalers):
    """Predicts the fitness value using the Kriging model (without uncertainty)."""
    X_new = np.array(individual).reshape(1, -1)
    # 1. Scale input X
    X_scaled = (X_new - scalers['X_min']) / (scalers['X_max'] - scalers['X_min'])
    # 2. Predict scaled log Y
    Y_mean_scaled = model.predict(X_scaled, return_std=False)
    Y_mean_scaled = np.ravel(Y_mean_scaled)[0]
    # 3. Inverse Log Transformation (Y_real = exp(Y_scaled * Y_std + Y_mean))
    predicted_fitness = np.exp(Y_mean_scaled * scalers['Y_std'] + scalers['Y_mean'])
    # Ensure fitness is non-negative (can happen due to prediction errors on log scale)
    return max(predicted_fitness, 0.1)

def normalize_data(X, Y):
    # Min-Max Scaling for X
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_scaled = (X - X_min) / (X_max - X_min)
    # Logarithmic Transformation for Y
    Y_log = np.log(Y)
    # Standard Scaling for Log(Y)
    Y_mean, Y_std = Y_log.mean(), Y_log.std()
    Y_scaled = (Y_log - Y_mean) / Y_std
    scalers = {'X_min': X_min, 'X_max': X_max, 'Y_mean': Y_mean, 'Y_std': Y_std}
    return X_scaled, Y_scaled, scalers

def generate_dataset(num_samples=1000, filename=DATASET_FILENAME):
    data = []
    # FEASIBLE SEEDING
    seed_individual = [5.0, 10.0, 12.0, 8.0, 7.0, 9.0, 11.0, 6.0, 10.0, 13.0]
    seed_fitness = calculate_fitness(seed_individual)
    data.append(seed_individual + [seed_fitness])
    print(f"SEED ADDED: Fitness = {seed_fitness:.4e}")

    print(f"Generating dataset ('{filename}') with {num_samples} samples...")
    for i in range(num_samples):
        individual = [random.uniform(3.0, 20.0) for _ in range(10)]
        fitness = calculate_fitness(individual)
        data.append(individual + [fitness])
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1} samples...")

    columns = [f'A{i+1}' for i in range(10)] + ['Fitness']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
    print(f"Dataset generated and saved to '{filename}'.")
    return filename

def train_kriging_model(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :NUM_DESIGN_VARIABLES].values
    Y = df['Fitness'].values.reshape(-1, 1)
    X_scaled, Y_scaled, scalers = normalize_data(X, Y)
    # Split for model validation check
    X_train, _, Y_train, _ = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)
    print("\nStarting Kriging Model Training...")
    # Model Training
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, length_scale_bounds='fixed')
    krige = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=3, alpha=1e-6, random_state=42
    )
    krige.fit(X_train, Y_train)
    print("✅ Kriging Model Trained Successfully.")
    return krige, scalers

## ----------------------------------------------------------------------------- 
## 2. PSO ALGORITHMS (True vs. Surrogate-Assisted)
## ----------------------------------------------------------------------------- 

def initialize_swarm(pop_size):
    swarm = []
    for _ in range(pop_size):
        position = [random.uniform(A_MIN, A_MAX) for _ in range(10)]
        velocity = [random.uniform(-1.0, 1.0) for _ in range(10)]
        particle = {
            'position': position,
            'velocity': velocity,
            'pbest_position': position[:],
            'pbest_value': float('inf'),
            'fitness': float('inf')}
        swarm.append(particle)
    return swarm

def pso_true_fitness_algorithm(max_fe_limit, pop_size):
    w = 0.7
    c1 = 2.0
    c2 = 2.0

    swarm = initialize_swarm(pop_size)
    gbest_value = float('inf')
    gbest_position = None

    fe_count = 0

    # INITIAL EVALUATION
    for particle in swarm:
        if fe_count >= max_fe_limit: break
        particle_fitness = calculate_fitness(particle['position'])
        fe_count += 1
        particle['fitness'] = particle_fitness
        particle['pbest_value'] = particle_fitness
        particle['pbest_position'] = particle['position'][:]
        if particle_fitness < gbest_value:
            gbest_value = particle_fitness
            gbest_position = particle['position'][:]

    iteration = 0
    while fe_count < max_fe_limit:
        if gbest_position is None:
            # safety: if initial evals failed
            gbest_position = swarm[0]['position'][:]
        for particle in swarm:
            if fe_count >= max_fe_limit: break
            for i in range(10):
                r1 = random.random()
                r2 = random.random()
                new_velocity = (w * particle['velocity'][i] +
                                c1 * r1 * (particle['pbest_position'][i] - particle['position'][i]) +
                                c2 * r2 * (gbest_position[i] - particle['position'][i]))
                particle['velocity'][i] = new_velocity
                new_position = particle['position'][i] + new_velocity
                particle['position'][i] = np.clip(new_position, A_MIN, A_MAX)

            new_fitness = calculate_fitness(particle['position'])
            fe_count += 1
            if new_fitness < particle['pbest_value']:
                particle['pbest_value'] = new_fitness
                particle['pbest_position'] = particle['position'][:]
            if new_fitness < gbest_value:
                gbest_value = new_fitness
                gbest_position = particle['position'][:]
        iteration += 1

    return gbest_value, fe_count, iteration

def pso_surrogate_assisted_algorithm(max_fe_limit, pop_size, kriging_model, scalers, max_iterations=5000):
    """
    SA-PSO with an iteration safety cap to prevent endless loops if fe_count stagnates.
    """
    w = 0.7
    c1 = 2.0
    c2 = 2.0
    
    swarm = initialize_swarm(pop_size)
    gbest_value = float('inf')
    gbest_position = None
    fe_count = 0
    
    # --- 1. INITIAL EVALUATION (TRUE FITNESS) ---
    for particle in swarm:
        if fe_count >= max_fe_limit: break
        particle_fitness = calculate_fitness(particle['position'])
        fe_count += 1
        
        particle['fitness'] = particle_fitness
        particle['pbest_value'] = particle_fitness
        particle['pbest_position'] = particle['position'][:]
        
        if particle_fitness < gbest_value:
            gbest_value = particle_fitness
            gbest_position = particle['position'][:]

    # --- 2. MAIN OPTIMIZATION LOOP ---
    iteration = 0
    
    # Dış döngü koşulu: Hem FE limitimiz dolmadıysa hem de iterasyon limitini aşmadıysak devam et.
    while fe_count < max_fe_limit and iteration < max_iterations: 
        
        if gbest_position is None:
             gbest_position = swarm[0]['position'][:]

        for particle in swarm:
            # İç döngüde kontrol: Limit dolarsa hemen çık.
            if fe_count >= max_fe_limit: break 

            # 1. Pozisyon ve Hız Güncelleme (Aynı Kalır)
            for i in range(10):
                r1, r2 = random.random(), random.random()
                new_velocity = (w * particle['velocity'][i] + 
                                c1 * r1 * (particle['pbest_position'][i] - particle['position'][i]) + 
                                c2 * r2 * (gbest_position[i] - particle['position'][i]))
                particle['velocity'][i] = new_velocity
                new_position = np.clip(particle['position'][i] + new_velocity, A_MIN, A_MAX)
                particle['position'][i] = new_position

            # 2. KRIGING TAHMİNİ 
            new_fitness_surrogate = kriging_predict_value(kriging_model, particle['position'], scalers)
            
            # 3. DOĞRULAMA (TRUE FE KULLANIMI)
            if new_fitness_surrogate < particle['pbest_value']:
                
                # True Fitness çağrılır ve FE Sayımı artar.
                true_fitness = calculate_fitness(particle['position'])
                fe_count += 1
                
                # 4. GÜNCELLEMELER (TRUE FITNESS KULLANARAK)
                if true_fitness < particle['pbest_value']:
                    particle['pbest_value'] = true_fitness
                    particle['pbest_position'] = particle['position'][:]

                    if true_fitness < gbest_value:
                        gbest_value = true_fitness
                        gbest_position = particle['position'][:]
            
        iteration += 1

    return gbest_value, fe_count, iteration
## ----------------------------------------------------------------------------- 
## 3. MAIN EXECUTION
## ----------------------------------------------------------------------------- 

if __name__ == "__main__":
    # --- KRIGING MODEL PREPARATION (SECTION A) ---
    stable_filename = generate_dataset(num_samples=1000)
    kriging_model, scalers = train_kriging_model(stable_filename)

    print("\n" + "="*50)
    print("STARTING SURROGATE-ASSISTED PSO (SA-PSO)")
    print("="*50)

    POP_SIZE = 50
    FE_LIMIT_1000 = 1000
    FE_LIMIT_5000 = 5000

    # --- RUN 1: 1000 RUNS (TRUE PSO) ---
    print("\n--- 1000 Runs: Pure PSO (Baseline) ---")
    start_time_pso_1000 = time.time()
    best_fitness_pso_1000, fe_count_pso_1000, iterations_pso_1000 = pso_true_fitness_algorithm(FE_LIMIT_1000, POP_SIZE)
    total_time_pso_1000 = time.time() - start_time_pso_1000

    print(f"Best Fitness (Pure PSO): {best_fitness_pso_1000:.4f}")
    print(f"Total True FE: {fe_count_pso_1000}")
    print(f"Total Time: {total_time_pso_1000:.4f} seconds")

    # --- RUN 2: 1000 RUNS (SA-PSO) ---
    print("\n--- 1000 Runs: Surrogate-Assisted PSO (SA-PSO) ---")
    start_time_sapso_1000 = time.time()
    best_fitness_sapso_1000, fe_count_sapso_1000, iterations_sapso_1000 = pso_surrogate_assisted_algorithm(
        FE_LIMIT_1000, POP_SIZE, kriging_model, scalers, max_iterations=5000) 
    total_time_sapso_1000 = time.time() - start_time_sapso_1000
    
    print(f"Best Fitness (SA-PSO): {best_fitness_sapso_1000:.4f}")
    print(f"Total True FE: {fe_count_sapso_1000}")
    print(f"Total Iterations (Kriging used): {iterations_sapso_1000}")
    print(f"Total Time: {total_time_sapso_1000:.4f} seconds")
    
    # --- RUN 3: 5000 RUNS (SA-PSO) ---
    print("\n--- 5000 Runs: Surrogate-Assisted PSO (SA-PSO) ---")
    start_time_sapso_5000 = time.time()
    best_fitness_sapso_5000, fe_count_sapso_5000, iterations_sapso_5000 = pso_surrogate_assisted_algorithm(
        FE_LIMIT_5000, POP_SIZE, kriging_model, scalers, max_iterations=25000) 
    total_time_sapso_5000 = time.time() - start_time_sapso_5000
    
    print(f"Best Fitness (SA-PSO): {best_fitness_sapso_5000:.4f}")
    print(f"Total True FE: {fe_count_sapso_5000}")
    print(f"Total Iterations (Kriging used): {iterations_sapso_5000}")
    print(f"Total Time: {total_time_sapso_5000:.4f} seconds")

    # --- RUN 4: 5000 Runs (PURE PSO) ---
    print("\n--- 5000 Runs: Pure PSO (Baseline) ---")
    start_time_pso_5000 = time.time()
    best_fitness_pso_5000, fe_count_pso_5000, iterations_pso_5000 = pso_true_fitness_algorithm(FE_LIMIT_5000, POP_SIZE)
    total_time_pso_5000 = time.time() - start_time_pso_5000
    
    print(f"Best Fitness (Pure PSO 5000 FE): {best_fitness_pso_5000:.4f}")
    print(f"Total True FE: {fe_count_pso_5000}")
    print(f"Total Time: {total_time_pso_5000:.4f} seconds")


    # --- SUMMARY ---
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"Pure PSO (1000 FE): Fitness={best_fitness_pso_1000:.4f}, Time={total_time_pso_1000:.4f}s")
    print(f"SA-PSO (1000 FE): Fitness={best_fitness_sapso_1000:.4f}, Time={total_time_sapso_1000:.4f}s")
    print(f"Pure PSO (5000 FE): Fitness={best_fitness_pso_5000:.4f}, Time={total_time_pso_5000:.4f}s")
    print(f"SA-PSO (5000 FE): Fitness={best_fitness_sapso_5000:.4f}, Time={total_time_sapso_5000:.4f}s")
