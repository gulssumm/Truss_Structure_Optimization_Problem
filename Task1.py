import random

# Generates the initial population for the 10-Bar Truss
# Initialization
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

# Selection
def tournament_selection(population):
    tournament_pool = random.sample(population, 5)
    best_individual = tournament_pool[0]
    best_fitness = calculate_fitness(best_individual)

    for i in range(len(tournament_pool)):
        current_individual = tournament_pool[i]
        current_fitness = calculate_fitness(current_individual)

        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_individual = current_individual

    return best_individual

# Crossover
def crossover(parent1, parent2):
    if random.random() <= 0.3:  # crossover possibility 30%
        cut_point = random.randint(1, 9)              # random index
        child1 = parent1[:cut_point] + parent2[cut_point:]
        child2 = parent2[:cut_point] + parent1[cut_point:]
        return child1, child2
    else:
        return parent1, parent2

if __name__ == "__main__":
    # 1.Initialize
    population = initialize_population()

    # 2.Tournament
    parent = tournament_selection(population)
    print("Selected parent fitness: ", calculate_fitness(parent))

    # 3.Crossover
