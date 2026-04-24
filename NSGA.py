import numpy as np
import random

inf = 1000000000
RHO = 1
KAPPA = 2
N_sensor = 100
N = 100
gen = 10001
p_mutation = 0.3
barrierLength = 1000

outputFile = "result_NSGA.csv"
inputFile = "dataset/100_1.txt"
X = np.loadtxt(inputFile, dtype=int)


class Individual:
    def __init__(self, f1=0, f2=0, f3=0):
        self.s = []
        # [0, 1, 0, 0, ...]
        self.r = []
        # [10.5, 30,0, 0, 23.0, ...]
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.rank = 0
        self.crowding_distance = 0


def initIndividual():
    individual = []
    for _ in range(N_sensor):
        individual.append(random.randint(0, 1))
    return individual


def is_all_zero(array):
    for i in array:
        if i != 0:
            return False
    return True


def initValidIndividual():
    individual = initIndividual()
    while is_all_zero(individual):
        individual = initIndividual()
    return individual


def coverShrink(individual):
    index = [i for i in range(N_sensor) if individual[i] == 1]
    k = len(index)

    half_dis = [X[index[0]], *[(X[index[i+1]] - X[index[i]]) / 2 for i in range(k-1)], barrierLength - X[index[-1]]]

    r = [max(half_dis[0], half_dis[1]), max(half_dis[k-1], half_dis[k])]
    for i in range(1, k-1):
        r.insert(-1, max(half_dis[i], half_dis[i+1]))

    c_r = []
    r_index = 0
    for i in range(N_sensor):
        if individual[i] == 1:
            c_r.append(r[r_index])
            r_index += 1
        else:
            c_r.append(0)
    # Ex: c_r = [0, 100, 0, 0, 59, 74, 0,]; r = [100, 59, 74]

    # shrink
    shrink_iterations = 0
    while shrink_iterations < 1:
        for i in range(1, k-1):
            curr_left = X[index[i]] - r[i]
            curr_right = X[index[i]] + r[i]
            prev_left = X[index[i-1]] - r[i-1]
            prev_right = X[index[i-1]] + r[i-1]
            next_left = X[index[i+1]] - r[i+1]
            next_right = X[index[i+1]] + r[i+1]

            if (prev_right >= curr_right):
                r[i] = 0
                c_r[index[i]] = 0
                individual[index[i]] = 0

            if (next_left <= curr_left):
                r[i] = 0
                c_r[index[i]] = 0
                individual[index[i]] = 0

            if (curr_left <= prev_left):
                r[i-1] = 0
                c_r[index[i-1]] = 0
                individual[index[i-1]] = 0

            if (curr_right >= next_right):
                r[i+1] = 0
                c_r[index[i+1]] = 0
                individual[index[i+1]] = 0

            if (curr_left < prev_right) and (curr_right > next_left):
                c_r[index[i]] = max(0, max(
                    X[index[i]] - X[index[i-1]] - r[i-1], X[index[i+1]] - X[index[i]] - r[i+1]))
                if c_r[index[i]] == 0:
                    r[i] = 0
                    individual[index[i]] = 0
                else:
                    r[i] = c_r[index[i]]
        shrink_iterations += 1
    # print(individual)
    return c_r


def evaluate(r_i):
    f1, f2, f3 = 0, 0, 0
    for i in range(N_sensor):
        if r_i[i] > 0:
            f2 += 1
            f1 += RHO * (r_i[i] ** KAPPA)
        if r_i[i] > f3:
            f3 = r_i[i]

    return f1, f2, f3


def mutation(individual):
    newIndividual = individual
    while True:
        index = random.randint(0, N_sensor-1)
        if newIndividual[index] == 1:
            newIndividual[index] = 0
            break
    while True:
        index = random.randint(0, N_sensor-1)
        if newIndividual[index] == 0:
            newIndividual[index] = 1
            break
    return newIndividual


def mutation2(individual):
    newIndividual = individual
    for _ in range(4):
        index = random.randint(0, N_sensor-1)
        newIndividual[index] = (int(newIndividual[index]) + 1) % 2
    return newIndividual


def crossover(parent1, parent2):
    index = random.randint(1, N_sensor-1)
    child1 = parent1[:index] + parent2[index:]
    child2 = parent2[:index] + parent1[index:]

    while is_all_zero(child1):
        index = random.randint(1, N_sensor-1)
        child1 = parent1[:index] + parent2[index:]
    while is_all_zero(child2):
        index = random.randint(1, N_sensor-1)
        child2 = parent2[:index] + parent1[index:]

    if random.random() < p_mutation:
        child1 = mutation(child1)
    if random.random() < p_mutation:
        child2 = mutation(child2)

    return child1, child2


def crossover2(parent1, parent2):
    gap = 5
    child1, child2 = [], []

    for i in range(0, N_sensor, gap):
        rand = random.random()
        if rand > 0.5:
            child1 += parent1[i:i+gap]
            child2 += parent2[i:i+gap]
        else:
            child1 += parent2[i:i+gap]
            child2 += parent1[i:i+gap]

    if random.random() < p_mutation:
        child1 = mutation2(child1)
    if random.random() < p_mutation:
        child2 = mutation2(child2)

    return child1, child2


def non_dominated_rank(population):
    # N = 5
    pop_size = N*2
    ranks = [0] * pop_size
    dominating_list = [[] for _ in range(pop_size)]
    dominated_count = [0] * pop_size
    for i in range(pop_size):
        for j in range(i+1, pop_size):
            if (population[i].f1 <= population[j].f1 and population[i].f2 < population[j].f2 and population[i].f3 < population[j].f3) or (population[i].f1 < population[j].f1 and population[i].f2 <= population[j].f2 and population[i].f3 < population[j].f3) or (population[i].f1 < population[j].f1 and population[i].f2 < population[j].f2 and population[i].f3 <= population[j].f3):
                dominating_list[i].append(j)
                dominated_count[j] += 1
            elif (population[i].f1 >= population[j].f1 and population[i].f2 > population[j].f2 and population[i].f3 > population[j].f3) or (population[i].f1 > population[j].f1 and population[i].f2 >= population[j].f2 and population[i].f3 > population[j].f3) or (population[i].f1 > population[j].f1 and population[i].f2 > population[j].f2 and population[i].f3 >= population[j].f3):
                dominating_list[j].append(i)
                dominated_count[i] += 1

    current_rank = 0
    current_front = [i for i in range(pop_size) if dominated_count[i] == 0]

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


def crowding_distance(population):
    N = len(population)
    for i in range(N):
        population[i].crowding_distance = 0

    population.sort(key=lambda x: x.f1)
    population[0].crowding_distance = inf
    population[-1].crowding_distance = inf
    for i in range(1, N-1):
        population[i].crowding_distance += (population[i+1].f1 - population[i-1].f1) / \
            (population[-1].f1 - population[0].f1)

    population.sort(key=lambda x: x.f2)
    population[0].crowding_distance = inf
    population[-1].crowding_distance = inf
    for i in range(1, N-1):
        population[i].crowding_distance += (population[i+1].f2 - population[i-1].f2) / \
            (population[-1].f2 - population[0].f2)

    population.sort(key=lambda x: x.f3)
    population[0].crowding_distance = inf
    population[-1].crowding_distance = inf
    for i in range(1, N-1):
        population[i].crowding_distance += (population[i+1].f3 - population[i-1].f3) / \
            (population[-1].f3 - population[0].f3)

    population.sort(key=lambda x: x.crowding_distance, reverse=True)

    return population


with open(outputFile, "w") as file:
    file.write(
        "Gen:        Individual:       Power:        Sensors:      Fairness:\n")

population = []

# init individual
for i in range(N):
    individual = Individual()
    individual.s = initValidIndividual()
    individual.r = coverShrink(individual.s)
    individual.f1, individual.f2, individual.f3 = evaluate(individual.r)
    population.append(individual)

for i in range(gen):
    # print("----")
    np.random.shuffle(population)
    for j in range(0, N, 2):
        # Crossover
        child1, child2 = Individual(), Individual()
        child1.s, child2.s = crossover2(population[j].s, population[j+1].s)
        while (is_all_zero(child1.s) or is_all_zero(child2.s)):
            child1.s, child2.s = crossover2(population[j].s, population[j+1].s)

        child1.r, child2.r = coverShrink(child1.s), coverShrink(child2.s)

        child1.f1, child1.f2, child1.f3 = evaluate(child1.r)
        child2.f1, child2.f2, child2.f3 = evaluate(child2.r)
        population.append(child1)
        population.append(child2)

    # fast non-dominated sorting
    ranks = non_dominated_rank(population)
    # print(ranks)
    for j in range(N*2):
        population[j].rank = ranks[j]
        # print(population[j].rank)

    # crowding distance
    pop_rank = 0
    next_population = []
    while (True):
        pop_rank_no = [population[k]
                       for k in range(N*2) if population[k].rank == pop_rank]
        required_size = N - len(next_population)
        if len(pop_rank_no) <= required_size:
            next_population += pop_rank_no
            pop_rank += 1
        else:
            pop_rank_no = crowding_distance(pop_rank_no)
            next_population += pop_rank_no[:required_size]
            break
        if (required_size == 0):
            break

    population = next_population
    print(i)
    for j in range(N):
        # if (j %1000 == 0):
        #     print(j,"\n", population[j].r)
        with open(outputFile, "a") as file:
            output_string = f"{{:<{13}}} {{:<{15}}} {{:<{
                15}}} {{:<{13}}} {{:<{13}}}\n"
            file.write(output_string.format(
                i+1, j+1, population[j].f1/1000, population[j].f2, population[j].f3))
    
for i in range(N):
    print(population[i].f1, population[i].f2, population[i].f3)
    print(population[i].r)
