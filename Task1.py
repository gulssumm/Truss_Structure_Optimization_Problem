import random
import numpy as np

# Generates the initial population for the 10-Bar Truss
def initialize_population(pop_size=50, num_vars=10, min_area=0.01, max_area=35.0):
    population = []  # population size = 50
    for _ in range(pop_size):
        individual = []
        # (A_i)
        for _ in range(num_vars):
            # Generate a random real number (float)
            area = random.uniform(min_area, max_area)
            individual.append(area)
        population.append(individual)

    return population


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