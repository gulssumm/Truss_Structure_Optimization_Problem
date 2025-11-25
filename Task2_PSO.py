import random, math, time

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

# Initialize
def initialize_swarm(pop_size):
    swarm = []
    for _ in range(pop_size):
        position = []
        velocity = []
        for i in range(10):
            position.append(random.uniform(0.01, 35.0))
            velocity.append(random.uniform(-1.0, 1.0))

        particle = {
        'position': position,
        'velocity': velocity,
        'pbest_position': position[:],
        'pbest_value': float('inf'),
        'fitness': float('inf')}

        swarm.append(particle)
    return swarm