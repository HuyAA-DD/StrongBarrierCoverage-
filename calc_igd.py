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


moead_igd_mean = []
nsga_igd_mean = []

for dataset in datasets:
    moead_igd_values = []
    nsga_igd_values = []

    for i in range(num_subdatasets):
        true_pareto_front = load_data(f"{base_link}/approx/{dataset}_{i}.csv")
        min_vals = np.min(true_pareto_front, axis=0)
        max_vals = np.max(true_pareto_front, axis=0)
        norm_true_pareto_front = normalize(true_pareto_front, min_vals, max_vals)

        for run in range(run_start, run_end):
            moead_pareto_front = load_data(
                f"{base_link}/moead/moead_{dataset}_{i}_{run}.csv"
            )
            nsga_pareto_front = load_data(
                f"{base_link}/nsga/nsga_{dataset}_{i}_{run}.csv"
            )

            norm_moead_pareto_front = normalize(moead_pareto_front, min_vals, max_vals)
            norm_nsga_pareto_front = normalize(nsga_pareto_front, min_vals, max_vals)

            moead_igd = calculate_igd(norm_moead_pareto_front, norm_true_pareto_front)
            nsga_igd = calculate_igd(norm_nsga_pareto_front, norm_true_pareto_front)

            moead_igd_values.append(moead_igd)
            nsga_igd_values.append(nsga_igd)

    moead_igd_mean.append(np.mean(moead_igd_values))
    nsga_igd_mean.append(np.mean(nsga_igd_values))

print("MOEAD IGD Mean:", moead_igd_mean)
print("NSGA IGD Mean:", nsga_igd_mean)

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
index = np.arange(len(datasets))

bar1 = ax.bar(index, moead_igd_mean, bar_width, label="MOEA/D", color="red")
bar2 = ax.bar(
    index + bar_width, nsga_igd_mean, bar_width, label="NSGA-II", color="blue"
)

ax.set_xlabel("Dataset", fontsize=14)
ax.set_ylabel("Mean IGD", fontsize=14)
ax.set_title("Comparison of IGD values for MOEA/D and NSGA-II", fontsize=16)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(datasets)
ax.legend(fontsize=14)

plt.show()
