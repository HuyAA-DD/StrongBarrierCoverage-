import numpy as np
import matplotlib.pyplot as plt

run_start = 0
run_end = 10
base_link = "./result/pareto/moead/moead_"
datasets = ["100_1", "150_1", "200_1", "250_1"]


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def spread_metric(pareto_front):
    pareto_front = sorted(pareto_front, key=lambda x: x[0])
    pareto_front = np.unique(np.array(pareto_front), axis=0)

    f1_max = pareto_front[-1][0]
    f2_max = pareto_front[-1][1]
    f1_min = pareto_front[0][0]
    f2_min = pareto_front[0][1]

    for point in pareto_front:
        point[0] = (point[0] - f1_min) / (f1_max - f1_min)
        point[1] = (point[1] - f2_min) / (f2_max - f2_min)

    distances = [
        euclidean_distance(pareto_front[i], pareto_front[i + 1])
        for i in range(1, len(pareto_front) - 2)
    ]

    d_f = euclidean_distance(pareto_front[0], pareto_front[1])
    d_l = euclidean_distance(pareto_front[-1], pareto_front[-2])

    d_avg = np.mean(distances)

    spread = (d_f + d_l + sum(abs(d - d_avg) for d in distances)) / (
        d_f + d_l + (len(pareto_front) - 1) * d_avg
    )

    return spread


spread_values_per_dataset = []

for dataset in datasets:
    spread_values = []
    for run in range(run_start, run_end):
        input_file = f"{base_link}{dataset}_{run}.csv"
        pareto_front = np.loadtxt(input_file, dtype=float, delimiter=",")
        spread_value = spread_metric(pareto_front)
        spread_values.append(spread_value)

    spread_values_per_dataset.append(spread_values)

plt.boxplot(
    spread_values_per_dataset, labels=["100", "150", "200", "250"], patch_artist=True
)

plt.ylabel("Spread Delta Value")
plt.xlabel("Number of Sensors")
plt.show()
