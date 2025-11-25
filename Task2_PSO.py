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

def pso_algorithm(max_iterations, pop_size):
    # X(t+1) = X(t) + V(t+1)
    # V(t+1) = wV(t) + c1×rand() × (Xpbest - X(t)) + c2×rand() × (Xgbest - X(t))
    w = 0.7  # wV(t)
    c1 = 2.0  # c1×rand() × (Xpbest - X(t))
    c2 = 2.0  # c2×rand() × (Xgbest - X(t))

    swarm = initialize_swarm(pop_size)

    gbest_value = float('inf')
    gbest_position = []
    for particle in swarm:
        particle_fitness = calculate_fitness(particle['position'])
        # Update particle's memory
        particle['fitness'] = particle_fitness
        particle['pbest_value'] = particle_fitness
        particle['pbest_position'] = particle['position'][:]
        # Update global best
        if particle_fitness < gbest_value:
            gbest_value = particle_fitness
            gbest_position = particle['position'][:]

    for iteration in range(max_iterations):
        for particle in swarm:
            for i in range(10):
                r1 = random.random()
                r2 = random.random()

                new_velocity = w*particle['velocity'][i] + c1*r1*(particle['pbest_position'][i] - particle['position'][i]) + c2*r2*(gbest_position[i] - particle['position'][i])

                particle['velocity'][i] = new_velocity

                new_position = particle['position'][i] + new_velocity

                if new_position < 0.01: new_position = 0.01
                if new_position > 35.0: new_position = 35.0
                particle['position'][i] = new_position

            new_fitness = calculate_fitness(particle['position'])
            # personel best
            if new_fitness < particle['pbest_value']:
                particle['pbest_value'] = new_fitness
                particle['pbest_position'] = particle['position'][:]

            # global best
            if new_fitness < gbest_value:
                gbest_value = new_fitness
                gbest_position = particle['position'][:]
    return gbest_value

if __name__ == "__main__":
    max_iterations = 1000
    pop_size = 50
    start_time = time.time()
    best_fitness = pso_algorithm(max_iterations, pop_size)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"PSO Results for {max_iterations} iterations:")
    print(f"Best Fitness: {best_fitness}")
    print(f"Total Time: {end_time - start_time:.4f} seconds")