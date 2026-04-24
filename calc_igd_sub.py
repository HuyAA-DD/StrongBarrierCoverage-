import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

run_start = 0
run_end = 10
num_subdatasets = 10
base_link = "./result/pareto"
datasets = ["100", "150", "200", "250"]

results = []

def calculate_igd(pareto_front, reference_front):
    distances = []
    for point in pareto_front:
        distances.append(np.min(np.linalg.norm(reference_front - point, axis=1)))
    return np.mean(distances)

def normalize(data, min_vals, max_vals):
    return (data - min_vals) / (max_vals - min_vals)

def load_data(file):
    data = pd.read_csv(file, header=None, delimiter=",")
    data.drop_duplicates(inplace=True)
    return data.values

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, dataset in enumerate(datasets):
    moead_igd_mean = []
    nsga_igd_mean = []

    for i in range(num_subdatasets):
        moead_igd_values = []
        nsga_igd_values = []

        true_pareto_front = load_data(f"{base_link}/approx/{dataset}_{i}.csv")
        min_vals = np.min(true_pareto_front, axis=0)
        max_vals = np.max(true_pareto_front, axis=0)
        normalized_true_pareto_front = normalize(true_pareto_front, min_vals, max_vals)

        for run in range(run_start, run_end):
            moead_pareto_front = load_data(f"{base_link}/moead/moead_{dataset}_{i}_{run}.csv")
            nsga_pareto_front = load_data(f"{base_link}/nsga/nsga_{dataset}_{i}_{run}.csv")

            normalized_moead_pareto_front = normalize(moead_pareto_front, min_vals, max_vals)
            normalized_nsga_pareto_front = normalize(nsga_pareto_front, min_vals, max_vals)

            moead_igd = calculate_igd(normalized_moead_pareto_front, normalized_true_pareto_front)
            nsga_igd = calculate_igd(normalized_nsga_pareto_front, normalized_true_pareto_front)

            moead_igd_values.append(moead_igd)
            nsga_igd_values.append(nsga_igd)

            print(f"Dataset: {dataset}, Subdataset: {i}, Run: {run}, MOEAD IGD: {moead_igd}, NSGA IGD: {nsga_igd}")

        moead_igd_mean.append(np.mean(moead_igd_values))
        nsga_igd_mean.append(np.mean(nsga_igd_values))

    ax = axes[idx]
    bar_width = 0.35
    index = np.arange(num_subdatasets)

    bar1 = ax.bar(index, moead_igd_mean, bar_width, label='MOEA/D', color='red')
    bar2 = ax.bar(index + bar_width, nsga_igd_mean, bar_width, label='NSGA-II', color='blue')

    ax.set_xlabel('Subdataset', fontsize=12)
    ax.set_ylabel('Mean IGD', fontsize=12)
    ax.set_title(f'Dataset {dataset}', fontsize=14)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([f"{i}" for i in range(num_subdatasets)])
    ax.legend()

plt.tight_layout()
plt.show()