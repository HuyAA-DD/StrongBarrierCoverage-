import sys
import math
import time
import random
import numpy as np
import os

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else:
    dataset = "100_0"

input_file = f"./dataset/{dataset}.txt"
print("-nspso", dataset)

x_corr = np.loadtxt(input_file, dtype=int)
num_sensor = len(x_corr)

INF = 10**18

# ====== Problem params ======
pop_size = 32
max_gen = 10000
k = 2
k_minus_1 = k - 1
beta = 1
gamma = 0.5
barrier_length = 1000

run_start = 0
run_end = 10

# ====== NSPSO params ======
archive_max_size = pop_size
w_max = 0.9
w_min = 0.4
c1 = 1.8
c2 = 1.8
v_max = 6.0
mutation_prob = 0.02

# ====== Cache ======
eval_cache = {}


# =========================================================
# Strong barrier coverage model
# =========================================================
def exp_approx_beta(term):
    beta_term = beta * term
    return 1 - beta_term + (beta_term ** 2) / 2 - (beta_term ** 3) / 6


def radius_formalize_outermost_sensor(x1, isFirst=True):
    if not isFirst:
        x1 = barrier_length - x1

    for r_u1_val in range(1, x1 + 1):
        term = x1 - k_minus_1 * r_u1_val
        if exp_approx_beta(term) >= gamma:
            return r_u1_val
    return x1


def radius_formalize_sensor(r_u1, x1, x2):
    if x1 + r_u1 * k_minus_1 >= x2:
        return 0

    k_minus_1_r_u1 = k_minus_1 * r_u1
    x1_certain = None

    for x in range(x1 + r_u1, x2 + 1):
        if exp_approx_beta(x - x1 - k_minus_1_r_u1) >= gamma:
            x1_certain = x
            break

    if x1_certain is None or x1_certain >= x2:
        return 0

    x1_certain = math.floor(x1_certain)
    r_u2_max = (x2 - x1_certain) / (k - 1)
    r_u2_min = (x2 - x1_certain) / k

    for r_u2_val in range(math.ceil(r_u2_min), math.ceil(r_u2_max) + 1):
        k_minus_1_r_u2 = k_minus_1 * r_u2_val
        x_min = (x1 + x2 - (k - 1) * (r_u1 - r_u2_val)) / 2
        expr_val = (
            exp_approx_beta(x_min - x1 - k_minus_1_r_u1)
            + exp_approx_beta(x2 - x_min - k_minus_1_r_u2)
            - gamma
        )
        if expr_val >= 0:
            return r_u2_val

    return 0


def radius_formalize(individual):
    active_idx = np.flatnonzero(individual).tolist()
    if not active_idx:
        return [0] * num_sensor

    r_u = [radius_formalize_outermost_sensor(x_corr[active_idx[0]])]
    r_0_count = 0

    for i in range(1, len(active_idx)):
        prev_idx = active_idx[i - 1 - r_0_count]
        curr_idx = active_idx[i]
        r_temp = radius_formalize_sensor(r_u[i - 1 - r_0_count], x_corr[prev_idx], x_corr[curr_idx])

        if r_temp == 0:
            r_0_count += 1
        else:
            r_0_count = 0
        r_u.append(r_temp)

    r_last = radius_formalize_outermost_sensor(x_corr[active_idx[-1]], isFirst=False)
    if r_last > r_u[-1]:
        r_u[-1] = r_last

    all_r_u = [0] * num_sensor
    for idx, rv in zip(active_idx, r_u):
        all_r_u[idx] = rv

    return all_r_u


def calc_energy_consumption(r_u):
    term1 = 0.5 * (k_minus_1 * r_u) ** 2
    exp_beta = math.exp(-beta * r_u)
    term2 = (1 - exp_beta * (1 + beta * r_u)) / (beta ** 2)
    term3 = (k_minus_1 * r_u * (1 - exp_beta)) / beta
    return term1 + term2 + term3


def evaluate(gene):
    key = tuple(int(x) for x in gene)
    cached = eval_cache.get(key)
    if cached is not None:
        return cached

    all_r_u = radius_formalize(gene)
    total_active_sensor = int(np.sum(gene))
    total_energy_consumption = 0.0
    for r in all_r_u:
        if r > 0:
            total_energy_consumption += calc_energy_consumption(r)

    result = (total_active_sensor, total_energy_consumption)
    eval_cache[key] = result
    return result


# =========================================================
# Utilities for multi-objective
# =========================================================
def dominates_obj(f1a, f2a, f1b, f2b):
    return ((f1a <= f1b and f2a < f2b) or
            (f1a < f1b and f2a <= f2b))


def sigmoid_array(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


class Particle:
    __slots__ = (
        "gene", "vel", "f1", "f2",
        "rank", "crowding_distance",
        "pbest_gene", "pbest_f1", "pbest_f2"
    )

    def __init__(self):
        self.gene = None
        self.vel = None
        self.f1 = None
        self.f2 = None
        self.rank = None
        self.crowding_distance = 0.0
        self.pbest_gene = None
        self.pbest_f1 = None
        self.pbest_f2 = None


class ArchiveItem:
    __slots__ = ("gene", "f1", "f2", "rank", "crowding_distance")

    def __init__(self, gene, f1, f2):
        self.gene = gene.copy()
        self.f1 = f1
        self.f2 = f2
        self.rank = 0
        self.crowding_distance = 0.0


# =========================================================
# Non-dominated sorting + crowding distance
# =========================================================
def non_dominated_rank(population):
    n = len(population)
    ranks = [0] * n
    dominating_list = [[] for _ in range(n)]
    dominated_count = [0] * n

    for i in range(n):
        pi = population[i]
        for j in range(i + 1, n):
            pj = population[j]
            if dominates_obj(pi.f1, pi.f2, pj.f1, pj.f2):
                dominating_list[i].append(j)
                dominated_count[j] += 1
            elif dominates_obj(pj.f1, pj.f2, pi.f1, pi.f2):
                dominating_list[j].append(i)
                dominated_count[i] += 1

    current_rank = 0
    current_front = [i for i in range(n) if dominated_count[i] == 0]

    while current_front:
        next_front = []
        for i in current_front:
            ranks[i] = current_rank
            for j in dominating_list[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        current_front = next_front
        current_rank += 1

    return ranks


def get_fronts(population):
    ranks = non_dominated_rank(population)
    fronts_dict = {}

    for ind, rank in zip(population, ranks):
        ind.rank = rank
        fronts_dict.setdefault(rank, []).append(ind)

    return [fronts_dict[r] for r in sorted(fronts_dict.keys())]


def calc_crowding_distance(front):
    n = len(front)
    if n == 0:
        return
    if n <= 2:
        for ind in front:
            ind.crowding_distance = INF
        return

    for ind in front:
        ind.crowding_distance = 0.0

    # objective f1
    sorted_f1 = sorted(front, key=lambda x: x.f1)
    sorted_f1[0].crowding_distance = INF
    sorted_f1[-1].crowding_distance = INF
    f1_min, f1_max = sorted_f1[0].f1, sorted_f1[-1].f1
    if f1_max != f1_min:
        denom = f1_max - f1_min
        for i in range(1, n - 1):
            sorted_f1[i].crowding_distance += (sorted_f1[i + 1].f1 - sorted_f1[i - 1].f1) / denom

    # objective f2
    sorted_f2 = sorted(front, key=lambda x: x.f2)
    sorted_f2[0].crowding_distance = INF
    sorted_f2[-1].crowding_distance = INF
    f2_min, f2_max = sorted_f2[0].f2, sorted_f2[-1].f2
    if f2_max != f2_min:
        denom = f2_max - f2_min
        for i in range(1, n - 1):
            sorted_f2[i].crowding_distance += (sorted_f2[i + 1].f2 - sorted_f2[i - 1].f2) / denom


# =========================================================
# Archive (EP) update
# =========================================================
def update_archive(archive, candidates, archive_max_size):
    # deduplicate bằng dict theo gene tuple
    uniq = {}

    for item in archive:
        uniq[tuple(item.gene.tolist())] = ArchiveItem(item.gene, item.f1, item.f2)

    for obj in candidates:
        key = tuple(obj.gene.tolist())
        if key not in uniq:
            uniq[key] = ArchiveItem(obj.gene, obj.f1, obj.f2)

    combined = list(uniq.values())
    if not combined:
        return []

    fronts = get_fronts(combined)

    new_archive = []
    for front in fronts:
        calc_crowding_distance(front)
        if len(new_archive) + len(front) <= archive_max_size:
            new_archive.extend(front)
        else:
            front.sort(key=lambda x: x.crowding_distance, reverse=True)
            remain = archive_max_size - len(new_archive)
            new_archive.extend(front[:remain])
            break

    return new_archive


# =========================================================
# PSO-specific
# =========================================================
def random_nonzero_gene():
    gene = np.random.randint(0, 2, size=num_sensor, dtype=np.int8)
    while np.sum(gene) == 0:
        gene = np.random.randint(0, 2, size=num_sensor, dtype=np.int8)
    return gene


def init_swarm(pop_size):
    swarm = []
    for _ in range(pop_size):
        p = Particle()
        p.gene = random_nonzero_gene()
        p.vel = np.random.uniform(-1.0, 1.0, size=num_sensor)
        p.f1, p.f2 = evaluate(p.gene)

        p.pbest_gene = p.gene.copy()
        p.pbest_f1 = p.f1
        p.pbest_f2 = p.f2
        swarm.append(p)
    return swarm


def choose_leader(archive):
    if len(archive) == 1:
        return archive[0]
    a, b = random.sample(archive, 2)
    if a.crowding_distance > b.crowding_distance:
        return a
    if b.crowding_distance > a.crowding_distance:
        return b
    return a if random.random() < 0.5 else b


def maybe_mutate_gene(gene, prob=mutation_prob):
    mask = np.random.rand(num_sensor) < prob
    gene = gene.copy()
    gene[mask] = 1 - gene[mask]

    if np.sum(gene) == 0:
        gene[random.randint(0, num_sensor - 1)] = 1
    return gene


def update_particle(particle, leader, inertia_w):
    r1 = np.random.rand(num_sensor)
    r2 = np.random.rand(num_sensor)

    x = particle.gene.astype(float)
    pbest = particle.pbest_gene.astype(float)
    gbest = leader.gene.astype(float)

    new_vel = (
        inertia_w * particle.vel
        + c1 * r1 * (pbest - x)
        + c2 * r2 * (gbest - x)
    )
    new_vel = np.clip(new_vel, -v_max, v_max)

    prob_1 = sigmoid_array(new_vel)
    new_gene = (np.random.rand(num_sensor) < prob_1).astype(np.int8)
    new_gene = maybe_mutate_gene(new_gene)

    particle.vel = new_vel
    particle.gene = new_gene
    particle.f1, particle.f2 = evaluate(new_gene)


def update_pbest(particle):
    curr_better = dominates_obj(
        particle.f1, particle.f2,
        particle.pbest_f1, particle.pbest_f2
    )
    pbest_better = dominates_obj(
        particle.pbest_f1, particle.pbest_f2,
        particle.f1, particle.f2
    )

    if curr_better:
        particle.pbest_gene = particle.gene.copy()
        particle.pbest_f1 = particle.f1
        particle.pbest_f2 = particle.f2
    elif not pbest_better and random.random() < 0.5:
        particle.pbest_gene = particle.gene.copy()
        particle.pbest_f1 = particle.f1
        particle.pbest_f2 = particle.f2


# =========================================================
# Main
# =========================================================
def ensure_dirs():
    os.makedirs("./result/r/nspso", exist_ok=True)
    os.makedirs("./result/f/nspso", exist_ok=True)
    os.makedirs("./result/pareto/nspso", exist_ok=True)


def main():
    ensure_dirs()

    for run in range(run_start, run_end):
        path = f"./result/r/nspso/nspso_{dataset}_{run}.txt"
        if os.path.exists(path):
            print("Skip", dataset, run)
            continue

        print("-nspso run", run)
        time_start = time.time()

        archive_f = []
        swarm = init_swarm(pop_size)

        init_candidates = []
        for p in swarm:
            init_candidates.append(p)
            init_candidates.append(ArchiveItem(p.pbest_gene, p.pbest_f1, p.pbest_f2))

        archive = update_archive([], init_candidates, archive_max_size)

        for generation in range(max_gen):
            if (generation + 1) % 100 == 0:
                print("-nspso", generation)

            inertia_w = w_max - (w_max - w_min) * (generation / max_gen)

            if archive:
                calc_crowding_distance(archive)

            for p in swarm:
                leader = choose_leader(archive)
                update_particle(p, leader, inertia_w)
                update_pbest(p)

            candidates = []
            for p in swarm:
                candidates.append(p)
                candidates.append(ArchiveItem(p.pbest_gene, p.pbest_f1, p.pbest_f2))

            archive = update_archive(archive, candidates, archive_max_size)
            archive_f.extend((item.f1, item.f2) for item in archive)

        with open(f"./result/r/nspso/nspso_{dataset}_{run}.txt", "w") as file:
            for item in archive:
                r = radius_formalize(item.gene)
                file.write(f"{r}\n")

        with open(f"./result/f/nspso/nspso_{dataset}_{run}.csv", "w") as file:
            if archive_f:
                file.write(f"{max(x[0] for x in archive_f)} {max(x[1] for x in archive_f)}\n")
                for item in archive_f:
                    file.write(f"{item[0]} {item[1]}\n")
            else:
                file.write("0 0\n")

        with open("./result/nspso_time.csv", "a") as file:
            file.write(f"{pop_size}, {max_gen}, {dataset}, {run}, {time.time() - time_start}\n")

        with open(f"./result/pareto/nspso/nspso_{dataset}_{run}.csv", "w") as file:
            for item in archive:
                file.write(f"{item.f1}, {item.f2}\n")


if __name__ == "__main__":
    main()