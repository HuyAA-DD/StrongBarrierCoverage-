import sys
import math
import time
import random
import numpy as np
import os

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else:
    dataset = "300_0"

input_file = f"./dataset/{dataset}.txt"
print("-nsga", dataset)

x_corr = np.loadtxt(input_file, dtype=int)
num_sensor = len(x_corr)

inf = 99999
pop_size = 32 # Số lượng cá thể 
max_gen = 10000 # Số thế hệ tối đa
p_mutation = 0.3 # Hệ số tiến hóa 
k = 2 # R_s / R_u
k_minus_1 = k - 1
beta = 1 # Hệ số suy giảm 
gamma = 0.5 # detection threshold 
barrier_length = 1000 

run_start = 0
run_end = 10

lamb = []
z_nad = [None, None]


def mutation(gene):
    '''
    Đột biến gene
    Đảo bit ngẫu nhiên 10 nhiễm sắc thể 
    '''
    new_gene = gene[:]
    for _ in range(10):
        index = random.randint(0, num_sensor - 1)
        new_gene[index] ^= 1
    return new_gene


def crossover(parent1, parent2):
    '''
    lai ghép nhóm 2 bit liên tục trên cặp 
    cha mẹ parent1, parent2
    '''
    gap = 2
    child1, child2 = [], []

    for i in range(0, num_sensor, gap):
        if random.random() > 0.5:
            child1.extend(parent1[i : i + gap])
            child2.extend(parent2[i : i + gap])
        else:
            child1.extend(parent2[i : i + gap])
            child2.extend(parent1[i : i + gap])

    # mutation
    if random.random() < p_mutation:
        child1 = mutation(child1)
    if random.random() < p_mutation:
        child2 = mutation(child2)

    return child1, child2


def exp_approx_beta(term):
    # Taylor expansion for e^(-x): 1 - x + x^2 / 2! - x^3 / 3! + ...
    beta_term = beta * term
    return 1 - beta_term + (beta_term**2) / 2 - (beta_term**3) / 6


def radius_formalize_outermost_sensor(x1, isFirst=True):
    if not isFirst:
        x1 = barrier_length - x1

    min_r_u1 = x1
    for r_u1_val in range(1, x1 + 1):
        term = x1 - k_minus_1 * r_u1_val
        if exp_approx_beta(term) >= gamma:
            min_r_u1 = r_u1_val
            break

    return math.ceil(min_r_u1)


def radius_formalize_sensor(r_u1, x1, x2):
    if x1 + r_u1 * k_minus_1 >= x2:
        return 0

    k_minus_1_r_u1 = k_minus_1 * r_u1
    term = lambda x: x - x1 - k_minus_1_r_u1
    x1_certain = None
    for x in range(x1 + r_u1, x2 + 1):
        if exp_approx_beta(term(x)) >= gamma:
            x1_certain = x
            break

    if x1_certain >= x2:
        return 0

    x1_certain = math.floor(x1_certain)
    r_u2_max = (x2 - x1_certain) / (k - 1)
    r_u2_min = (x2 - x1_certain) / k

    valid_r_u2_val = None
    for r_u2_val in range(math.ceil(r_u2_min), math.ceil(r_u2_max) + 1):
        k_minus_1_r_u2 = k_minus_1 * r_u2_val
        # Tại x_min, giá trị tín hiệu là thấp nhất
        x_min = (x1 + x2 - (k - 1) * (r_u1 - r_u2_val)) / 2
        expr_val = (
            exp_approx_beta(x_min - x1 - k_minus_1_r_u1)
            + exp_approx_beta(x2 - x_min - k_minus_1_r_u2)
            - gamma
        )
        if expr_val >= 0:
            valid_r_u2_val = r_u2_val
            break

    # Trả về giá trị nhỏ nhất hợp lệ của r_u2_val
    if valid_r_u2_val is not None:
        return valid_r_u2_val

    return 0


def radius_formalize(individual):
    # Ex: all_r_u = [0, 100, 0, 0, 59, 74, 0,]; r_u = [100, 59, 74]
    index = [i for i in range(num_sensor) if individual[i] == 1]

    r_u, all_r_u, r_0_count = [], [], 0
    r_u.append(radius_formalize_outermost_sensor(x_corr[index[0]]))

    for i in range(1, len(index)):
        r_temp = radius_formalize_sensor(
            r_u[i - 1 - r_0_count], x_corr[index[i - 1 - r_0_count]], x_corr[index[i]]
        )
        if r_temp == 0:
            r_0_count += 1
        else:
            r_0_count = 0

        r_u.append(r_temp)

    r_last = radius_formalize_outermost_sensor(x_corr[index[-1]], isFirst=False)

    if r_last > r_u[-1]:
        r_u[-1] = r_last

    r_index = 0
    for i in range(num_sensor):
        if individual[i] == 1:
            all_r_u.append(r_u[r_index])
            r_index += 1
        else:
            all_r_u.append(0)

    return all_r_u


def calc_energy_consumption(r_u):
    '''
    Tính lượng năng lượng tiêu thụ với tập bán kính r_u
    '''
    term1 = 1 / 2 * (k_minus_1 * r_u) ** 2
    exp_beta = math.exp(-beta * r_u)
    term2 = (1 - exp_beta * (1 + beta * r_u)) / (beta**2)
    term3 = (k_minus_1 * r_u * (1 - exp_beta)) / beta
    total_energy_consumption = term1 + term2 + term3
    return total_energy_consumption


def evaluate(gene):
    all_r_u = radius_formalize(gene)
    total_active_sensor = sum(1 for g in gene if g > 0)
    total_energy_consumption = sum(calc_energy_consumption(r) for r in all_r_u if r > 0)

    return total_active_sensor, total_energy_consumption


class Individual:
    def __init__(self):
        self.f1 = None  # number of active sensor
        self.f2 = None  # total energy consumption
        self.r = None  # [100, 0, 59, 74, 0]
        self.gene = None  # [1, 0, 1, 1, 0]
        self.rank = None
        self.crowding_distance = None


def init_population(pop_size):
    '''
    Khởi tạo quần thể 
    '''
    population = []
    for _ in range(pop_size):
        individual = Individual()
        individual.gene = [random.randint(0, 1) for _ in range(num_sensor)]
        while sum(individual.gene) == 0:
            individual.gene = [random.randint(0, 1) for _ in range(num_sensor)]
        individual.f1, individual.f2 = evaluate(individual.gene)
        population.append(individual)
    return population


def non_dominated_rank(population):
    n_pop_size = len(population) # tại sao không dùng pop_size ?
    ranks = [0] * n_pop_size
    dominating_list = [[] for _ in range(n_pop_size)]
    dominated_count = [0] * n_pop_size

    for i in range(n_pop_size):
        for j in range(i + 1, n_pop_size):
            if (
                population[i].f1 <= population[j].f1
                and population[i].f2 < population[j].f2
            ) or (
                population[i].f1 < population[j].f1
                and population[i].f2 <= population[j].f2
            ):
                dominating_list[i].append(j)
                dominated_count[j] += 1
            elif (
                population[i].f1 >= population[j].f1
                and population[i].f2 > population[j].f2
            ) or (
                population[i].f1 > population[j].f1
                and population[i].f2 >= population[j].f2
            ):
                dominating_list[j].append(i)
                dominated_count[i] += 1

    current_rank = 0
    current_front = [i for i in range(n_pop_size) if dominated_count[i] == 0]

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


def calc_crowding_distance(population):
    n_pop_size = len(population)

    for ind in population:
        ind.crowding_distance = 0

    population.sort(key=lambda x: x.f1)
    population[0].crowding_distance = inf
    population[-1].crowding_distance = inf

    for i in range(1, n_pop_size - 1):
        population[i].crowding_distance += (
            population[i + 1].f1 - population[i - 1].f1
        ) / (population[-1].f1 - population[0].f1)

    population.sort(key=lambda x: x.f2)
    population[0].crowding_distance = inf
    population[-1].crowding_distance = inf

    for i in range(1, n_pop_size - 1):
        population[i].crowding_distance += (
            population[i + 1].f2 - population[i - 1].f2
        ) / (population[-1].f2 - population[0].f2)

    population.sort(key=lambda x: x.crowding_distance, reverse=True)

    return population


def main():
    for run in range(run_start, run_end, 1):

        path = f"./result/r/nsga/nsga_{dataset}_{run}.txt" #lấy đường dẫn
        if os.path.exists(path): #Bỏ qua dataset đã chạy rồi
            print("Skip", dataset, run)
            continue

        print("-nsga run", run)
        time_start = time.time()

        archive_f = []
        population = init_population(pop_size)

        for generation in range(max_gen):
            if (generation + 1) % 100 == 0:
                print("-nsga", generation)
            np.random.shuffle(population)

            for i in range(0, pop_size, 2):
                child1, child2 = Individual(), Individual()

                child1.gene, child2.gene = crossover(
                    population[i].gene, population[i + 1].gene
                )
                child1.f1, child1.f2 = evaluate(child1.gene)
                child2.f1, child2.f2 = evaluate(child2.gene)

                population.extend([child1, child2])

            ranks = non_dominated_rank(population)
            for i in range(len(population)):
                population[i].rank = ranks[i]

            rank = 0
            next_population = []

            while len(next_population) < pop_size:
                current_front = [ind for ind in population if ind.rank == rank]
                if len(current_front) + len(next_population) > pop_size:
                    current_front = calc_crowding_distance(current_front)
                next_population.extend(current_front[: pop_size - len(next_population)])
                rank += 1

            population = next_population
            archive_f.extend([[ind.f1, ind.f2] for ind in population])

        with open(f"./result/r/nsga/nsga_{dataset}_{run}.txt", "w") as file:
            pass

        for ind in population:
            r = radius_formalize(ind.gene)
            with open(f"./result/r/nsga/nsga_{dataset}_{run}.txt", "a") as file:
                file.write(f"{r}\n")

        with open(f"./result/f/nsga/nsga_{dataset}_{run}.csv", "w") as file:
            file.write(
                f"{max([pair[0] for pair in archive_f])} {max(pair[1] for pair in archive_f)}\n"
            )
            for item in archive_f:
                file.write(f"{item[0]} {item[1]}\n")

        with open("./result/nsga_time.csv", "a") as file:
            file.write(
                f"{pop_size}, {max_gen}, {dataset}, {run}, {time.time() - time_start}\n"
            )

        with open(f"./result/pareto/nsga/nsga_{dataset}_{run}.csv", "w") as file:
            for ind in population:
                file.write(f"{ind.f1}, {ind.f2}\n")


if __name__ == "__main__":
    main()
