import sys
import math
import time
import random
import numpy as np
import os # thêm 

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else:
    dataset = "100_1"

input_file = f"./dataset/{dataset}.txt"
print("moead", dataset)

x_corr = np.loadtxt(input_file, dtype=int)
num_sensor = len(x_corr)

inf = 99999
pop_size = 32 
max_gen = 10000 # Gốc là 10000
p_mutation = 0.3
neighbor_size = 6
k = 2
k_minus_1 = k - 1
beta = 1
gamma = 0.5
barrier_length = 1000

run_start = 0
run_end = 10 # gốc là 10

lamb = [] # lưu trữ các bài toán con 
z = [None, None] # z lí tưởng 
z_nad = [None, None] # z tệ nhất 


def init_lambda(pop_size): # khởi tạo cá bài toán con 
    step = 1 / (pop_size + 1) # chia trọng số đảm bảo phân phối đều 
    for i in range(1, pop_size + 1):
        a = i * step
        b = 1 - a
        lamb.append([round(a, 2), round(b, 2)])


init_lambda(pop_size)


def search_neighbor():
    lamb_array = np.array(lamb) # mảng pop_size * pop_size 
    distances = np.sqrt(
        ((lamb_array[:, np.newaxis, :] - lamb_array[np.newaxis, :, :]) ** 2).sum(axis=2) #khoảng cách Euclid giữa 2 bài toán 
    )

    np.fill_diagonal(distances, np.inf) # fill đường chéo = vô cùng 

    neighbors = [
        np.argsort(distances[i])[:neighbor_size].tolist() for i in range(pop_size) # Sắp xếp theo hàng xóm gần nhất 
    ]

    return neighbors


search_neighbor()


def mutation(gene):
    new_gene = gene[:]
    for _ in range(10):
        index = random.randint(0, num_sensor - 1)
        new_gene[index] ^= 1
    return new_gene


def crossover(parent1, parent2):
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
    k_minus_1 = k - 1
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

    return total_active_sensor, total_energy_consumption, all_r_u


def calc_fitness(f1, f2, z, z_nad, lamb_i):
    fitness = 0
    fitness += ((f1 - z[0]) / z_nad[0] - z[0]) / lamb_i[0]
    fitness += ((f2 - z[1]) / z_nad[1] - z[1]) / lamb_i[1]

    return fitness


class Individual: # Lớp biểu diễn cá thể 
    def __init__(self):
        self.f1 = None  # number of active sensor
        self.f2 = None  # total energy consumption
        self.r = None  # [100, 0, 59, 74, 0]
        self.gene = None  # [1, 0, 1, 1, 0]


def init_population(pop_size): #khởi tạo quần thể 
    population = []
    for _ in range(pop_size):
        individual = Individual()
        individual.gene = [random.randint(0, 1) for _ in range(num_sensor)] 
        while sum(individual.gene) == 0:
            individual.gene = [random.randint(0, 1) for _ in range(num_sensor)]
        individual.f1, individual.f2 = evaluate(individual.gene)[:2]  
        population.append(individual)
    return population


def main():
    for run in range(run_start, run_end, 1):
        
        path = f"./result/r/moead/moead_{dataset}_{run}.txt" #lấy đường dẫn
        if os.path.exists(path): #Bỏ qua dataset đã chạy rồi
            print("Skip", dataset, run)
            continue

        print("-moead run", run)
        time_start = time.time()

        archive_f = []
        neighbors = search_neighbor()
        population = init_population(pop_size)

        z = [
            min([ind.f1 for ind in population]),
            min([ind.f2 for ind in population]),
        ]
        z_nad = [
            max([ind.f1 for ind in population]),
            max([ind.f2 for ind in population]),
        ]

        for generation in range(max_gen):
            if (generation + 1) % 100 == 0:
                print("moead", generation)
            for i, ind in enumerate(population):
                lamb_i = lamb[i]
                ind_fit = calc_fitness(ind.f1, ind.f2, z, z_nad, lamb_i)
                neighbor_index = random.randint(0, neighbor_size - 1)
                neighbor = neighbors[i][neighbor_index]
                neighbor_gene = population[neighbor].gene

                child1, child2 = Individual(), Individual()
                child1.gene, child2.gene = crossover(ind.gene, neighbor_gene)

                child1.f1, child1.f2 = evaluate(child1.gene)[:2]
                child2.f1, child2.f2 = evaluate(child2.gene)[:2]

                child1_fit = calc_fitness(child1.f1, child1.f2, z, z_nad, lamb_i)
                child2_fit = calc_fitness(child2.f1, child2.f2, z, z_nad, lamb_i)

                min_fit = min(ind_fit, child1_fit, child2_fit)

                if min_fit == child1_fit:
                    population[i] = child1
                    z[0] = min(z[0], child1.f1)
                    z[1] = min(z[1], child1.f2)
                elif min_fit == child2_fit:
                    population[i] = child2
                    z[0] = min(z[0], child2.f1)
                    z[1] = min(z[1], child2.f2)

            z_nad[0] = max(ind.f1 for ind in population)
            z_nad[1] = max(ind.f2 for ind in population)

            archive_f.extend([[ind.f1, ind.f2] for ind in population]) #archive f nhét lần lượt các cá thể của từng thế hệ vào 

        with open(f"./result/r/moead/moead_{dataset}_{run}.txt", "w") as file:
            pass

        for ind in population:
            r = radius_formalize(ind.gene)
            with open(f"./result/r/moead/moead_{dataset}_{run}.txt", "a") as file:
                file.write(f"{r}\n") #bán kính của từng sensor trong pareto cuối cùng 

        with open(f"./result/f/moead/moead_{dataset}_{run}.csv", "w") as file:
            file.write(
                f"{max([pair[0] for pair in archive_f])} {max(pair[1] for pair in archive_f)}\n" #nadir point
            )
            for item in archive_f:
                file.write(f"{item[0]} {item[1]}\n") #vẽ phân bố pareto qua các thế hệ

        with open("./result/moead_time.csv", "a") as file:
            file.write(
                f"{pop_size}, {max_gen}, {dataset}, {run}, {time.time() - time_start}\n" #thời gian chạy với các tham số 
            )

        with open(f"./result/pareto/moead/moead_{dataset}_{run}.csv", "w") as file:
            for ind in population:
                file.write(f"{ind.f1}, {ind.f2}\n") #biên pareto cuối 


if __name__ == "__main__":
    main()
