import random, math, time

# Initialize
def initialize_population(bar_nums):
    population = []
    for _ in range(bar_nums):
        random_num = random.uniform(0.01,35.0)
        population.append(random_num)
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

# Neighbour
def get_neighbour(current_solution):
    cur_sol_list = current_solution.copy()
    i = random.randint(0,9)
    cur_sol_list[i] = random.uniform(0.01, 35.0)

    return cur_sol_list

def simulated_annealing(max_iterations):
    temperature = 10000000
    cooling_rate = 0.995

    current_solution = initialize_population(bar_nums=10)
    current_fitness = calculate_fitness(current_solution)

    best_solution = current_solution
    best_fitness = current_fitness

    for i in range(max_iterations):
        for i in range(10):
            neighbour_solution = get_neighbour(current_solution)
            neighbour_fitness = calculate_fitness(neighbour_solution)
            diff = neighbour_fitness - current_fitness
            if diff <= 0:                                  # current solution > neighbour solution (better)
                current_solution = neighbour_solution
                current_fitness = neighbour_fitness
            else:
                prob = math.exp(-diff / temperature)
                rand = random.random()
                if rand < prob:
                    current_solution = neighbour_solution
                    current_fitness = neighbour_fitness
            if neighbour_fitness < best_fitness:
                best_fitness = neighbour_fitness
                best_solution = current_solution[:]        # copy the list
        temperature *= cooling_rate
    return best_fitness

if __name__ == "__main__":
    start_time = time.time()
    best_fitness = simulated_annealing(5000)
    end_time = time.time()
    print(f"Best Fitness: {best_fitness} & Total Time: {end_time-start_time}")