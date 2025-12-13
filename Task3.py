
import random
import time
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split

# GLOBAL CONSTANTS
rho = 0.1
P = 10**6
C_stress = 0.08
C_disp = 0.25
NUM_DESIGN_VARIABLES = 10
DATASET_FILENAME = "truss_dataset.csv"
A_MIN = 0.01
A_MAX = 35.0
L_BARS = [360,360,360,360,510,510,510,510,510,510]


# TRUE FITNESS
def calculate_fitness(individual):
    w = sum(individual[i] * L_BARS[i] for i in range(NUM_DESIGN_VARIABLES))
    return w * rho + P*(C_stress + C_disp)


# KRIGING UTILITIES
def kriging_predict_value(model, individual, scalers):
    x = np.array(individual).reshape(1, -1)
    x_scaled = (x - scalers["X_min"]) / (scalers["X_max"] - scalers["X_min"])
    y_scaled = model.predict(x_scaled)[0]
    pred = np.exp(y_scaled * scalers["Y_std"] + scalers["Y_mean"])
    return max(pred, 0.1)

def normalize_data(X, Y):
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    Xs = (X - X_min) / (X_max - X_min)

    Y_log = np.log(Y)
    Y_mean = Y_log.mean()
    Y_std = Y_log.std()
    Ys = (Y_log - Y_mean) / Y_std

    return Xs, Ys, {"X_min": X_min, "X_max": X_max, "Y_mean": Y_mean, "Y_std": Y_std}

def generate_dataset(n=1000, filename=DATASET_FILENAME):
    data = []
    print(f"Generating dataset with {n} samples...")
    for _ in range(n):
        ind = [random.uniform(A_MIN, A_MAX) for _ in range(10)]
        fit = calculate_fitness(ind)
        data.append(ind + [fit])
    df = pd.DataFrame(data, columns=[f"A{i+1}" for i in range(10)] + ["Fitness"])
    df.to_csv(filename, index=False)
    return filename

def train_kriging_model(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :10].values
    Y = df["Fitness"].values.reshape(-1,1)

    Xs, Ys, scalers = normalize_data(X, Y)
    X_train, _, Y_train, _ = train_test_split(Xs, Ys, test_size=0.2, random_state=42)

    kernel = C(1.0) * RBF(1.0, length_scale_bounds="fixed")
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6, random_state=42)
    model.fit(X_train, Y_train)
    return model, scalers


# SA-GA
def init_ga(pop_size):
    return [[random.uniform(A_MIN, A_MAX) for _ in range(10)] for _ in range(pop_size)]

def tournament_surrogate(pop, model, scalers):
    candidates = random.sample(pop, 5)
    return min(candidates, key=lambda x: kriging_predict_value(model, x, scalers))

def crossover(p1, p2, rate=0.3):
    if random.random() <= rate:
        cp = random.randint(1, 9)
        return p1[:cp] + p2[cp:], p2[:cp] + p1[cp:]
    return p1[:], p2[:]

def mutate(ind, rate=0.1):
    for i in range(10):
        if random.random() <= rate:
            ind[i] = random.uniform(A_MIN, A_MAX)
    return ind

def worst_two_indices(pop, model, scalers):
    preds = [(i, kriging_predict_value(model, pop[i], scalers)) for i in range(len(pop))]
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[0][0], preds[1][0]

def sa_ga(max_fe, pop_size, model, scalers, max_gen=5000):
    pop = init_ga(pop_size)
    fe = 0
    true_cache = {}

    # Evaluate initial population
    for ind in pop:
        if fe >= max_fe: break
        fit = calculate_fitness(ind)
        true_cache[tuple(ind)] = fit
        fe += 1

    best_val = min(true_cache.values())
    best_ind = min(true_cache, key=true_cache.get)

    gen = 0
    while fe < max_fe and gen < max_gen:

        p1 = tournament_surrogate(pop, model, scalers)
        p2 = tournament_surrogate(pop, model, scalers)

        c1, c2 = crossover(p1, p2)
        c1, c2 = mutate(c1), mutate(c2)

        pred1 = kriging_predict_value(model, c1, scalers)
        pred2 = kriging_predict_value(model, c2, scalers)

        if pred1 < max(true_cache.values()) and fe < max_fe:
            t1 = calculate_fitness(c1); fe += 1
            true_cache[tuple(c1)] = t1
            if t1 < best_val: best_val, best_ind = t1, tuple(c1)

        if pred2 < max(true_cache.values()) and fe < max_fe:
            t2 = calculate_fitness(c2); fe += 1
            true_cache[tuple(c2)] = t2
            if t2 < best_val: best_val, best_ind = t2, tuple(c2)

        w1, w2 = worst_two_indices(pop, model, scalers)
        pop[w1], pop[w2] = c1, c2

        gen += 1

    return best_val, fe, gen

# SA-PSO
def init_swarm(pop_size):
    swarm = []
    for _ in range(pop_size):
        pos = [random.uniform(A_MIN, A_MAX) for _ in range(10)]
        vel = [random.uniform(-1, 1) for _ in range(10)]
        swarm.append({
            "position": pos,
            "velocity": vel,
            "pbest_position": pos[:],
            "pbest_value": float("inf"),
            "fitness": float("inf")
        })
    return swarm

def sa_pso(max_fe, pop_size, model, scalers, max_iter=5000):
    w, c1, c2 = 0.7, 2.0, 2.0
    swarm = init_swarm(pop_size)
    gbest_val = float("inf")
    gbest_pos = None
    fe = 0

    # Evaluate initial swarm
    for p in swarm:
        if fe >= max_fe: break
        t = calculate_fitness(p["position"])
        fe += 1

        p["fitness"] = t
        p["pbest_value"] = t
        p["pbest_position"] = p["position"][:]

        if t < gbest_val:
            gbest_val = t
            gbest_pos = p["position"][:]

    it = 0
    while fe < max_fe and it < max_iter:

        for p in swarm:
            if fe >= max_fe: break

            for i in range(10):
                r1, r2 = random.random(), random.random()
                v = (
                    w * p["velocity"][i] +
                    c1 * r1 * (p["pbest_position"][i] - p["position"][i]) +
                    c2 * r2 * (gbest_pos[i] - p["position"][i])
                )
                p["velocity"][i] = v
                new_pos = p["position"][i] + v
                p["position"][i] = np.clip(new_pos, A_MIN, A_MAX)

            pred = kriging_predict_value(model, p["position"], scalers)

            if pred < p["pbest_value"]:
                t = calculate_fitness(p["position"])
                fe += 1

                if t < p["pbest_value"]:
                    p["pbest_value"] = t
                    p["pbest_position"] = p["position"][:]

                if t < gbest_val:
                    gbest_val = t
                    gbest_pos = p["position"][:]

        it += 1

    return gbest_val, fe, it


# MAIN EXECUTION 
if __name__ == "__main__":

    # Generate dataset + train surrogate
    dataset = generate_dataset(1000)
    model, scalers = train_kriging_model(dataset)

    POP = 50
    FE_1000 = 1000
    FE_5000 = 5000

    print("\nSA-GA RUNS")

    t0 = time.time()
    best_ga_1000, fe_ga_1000, gen_ga_1000 = sa_ga(FE_1000, POP, model, scalers)
    t1 = time.time()
    print(f"SA-GA (1000 FE): best={best_ga_1000:.4f}, gen={gen_ga_1000}, time={t1-t0:.4f}s")

    t0 = time.time()
    best_ga_5000, fe_ga_5000, gen_ga_5000 = sa_ga(FE_5000, POP, model, scalers)
    t1 = time.time()
    print(f"SA-GA (5000 FE): best={best_ga_5000:.4f}, gen={gen_ga_5000}, time={t1-t0:.4f}s")

    print("\nSA-PSO RUNS")

    t0 = time.time()
    best_pso_1000, fe_pso_1000, it_pso_1000 = sa_pso(FE_1000, POP, model, scalers)
    t1 = time.time()
    print(f"SA-PSO (1000 FE): best={best_pso_1000:.4f}, it={it_pso_1000}, time={t1-t0:.4f}s")

    t0 = time.time()
    best_pso_5000, fe_pso_5000, it_pso_5000 = sa_pso(FE_5000, POP, model, scalers)
    t1 = time.time()
    print(f"SA-PSO (5000 FE): best={best_pso_5000:.4f}, it={it_pso_5000}, time={t1-t0:.4f}s")

    print("\nFINAL SUMMARY (ONLY SURROGATE VERSIONS)")

    print(f"SA-GA  (1000 FE): {best_ga_1000:.4f}")
    print(f"SA-GA  (5000 FE): {best_ga_5000:.4f}")
    print(f"SA-PSO (1000 FE): {best_pso_1000:.4f}")
    print(f"SA-PSO (5000 FE): {best_pso_5000:.4f}")
