import random
import pandas as pd

# Fitness Evaluation
def calculate_fitness(individual):
    
    # Calculates the fitness F(X) for a single individual
    # Constants from the assignment
    rho = 0.1  # Material density
    P = 10 ** 6  # Penalty constraint
    C_stress = 0.08  # Stress constraint value
    C_disp = 0.25  # Nodal displacement constraint value

    # Constant lengths for the 10 bars
    L = [360, 360, 360, 360, 510, 510, 510, 510, 510, 510]

    # 1. Calculate Total Weight W(X) = rho * sum(A_i * L_i)
    Sum_AiLi = 0
    for i in range(10):
        # A_i is the cross-sectional area from the individual
        A_i = individual[i]
        L_i = L[i]
        Sum_AiLi += (A_i * L_i)

    W_X = Sum_AiLi * rho

    # 2. Calculate Full Fitness F(X) = W + P * (C_stress + C_disp)
    Penalty_Term = P * (C_stress + C_disp)

    Fitness_F = W_X + Penalty_Term

    return Fitness_F

def generate_dataset(num_samples=1000):
    data = []

    for _ in range(num_samples):
        individual = []
        for i in range(10):
            individual.append(random.uniform(0.01, 35.0))

        fitness = calculate_fitness(individual)
        row = individual.copy()
        row.append(fitness)
        data.append(row)

    columns = [f'A{i+1}' for i in range(10)] + ['Fitness']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('truss_dataset.csv', index=False)
    print(f"Dataset generated with {num_samples} samples")

if __name__ == "__main__":
    generate_dataset()