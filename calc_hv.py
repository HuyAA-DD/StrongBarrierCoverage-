import numpy as np
import pandas as pd
from pygmo import hypervolume
import matplotlib.pyplot as plt

pop_size = 32
max_generation = 10000
dts = 200 #gốc là 250
datasets = [f"{dts}_{i}" for i in range(10)] #gốc là range 10
num_runs = 10


def load_data(file_name):
    data = pd.read_csv(file_name, header=None, sep="\s+")
    z_nad = data.iloc[0].to_numpy().astype(float)
    aaa = data.iloc[1:].values.astype(float).tolist()
    return z_nad, aaa


def calculate_hv(aaa, z_nad):
    hv_value = []
    for generation in range(max_generation):
        bbb = []
        for j in range(pop_size):
            idx = generation * pop_size + j
            if idx < len(aaa):
                bbb.append([aaa[idx][0] / z_nad[0], aaa[idx][1] / z_nad[1]])
        hv = hypervolume(bbb)
        hv_value.append(hv.compute([1, 1]))
    return hv_value


def calculate_statistics(filenames):
    all_hv_values = []
    for file_name in filenames:
        z_nad, aaa = load_data(file_name)
        hv_values = calculate_hv(aaa, z_nad)
        all_hv_values.append(hv_values)

    all_hv_values = np.array(all_hv_values)
    mean_hv = np.mean(all_hv_values, axis=0)
    std_hv = np.std(all_hv_values, axis=0)
    return mean_hv, std_hv


moead_files = [
    f"./result/f/moead/moead_{dataset}_{i}.csv" #f"./result/f/moead/{dts}/moead_{dataset}_{i}.csv"
    for dataset in datasets
    for i in range(num_runs)
]


nsga_files = [
    f"./result/f/nsga/nsga_{dataset}_{i}.csv" #f"./result/f/nsga/{dts}/nsga_{dataset}_{i}.csv"
    for dataset in datasets
    for i in range(num_runs)
]

mean_hv_moead, std_hv_moead = calculate_statistics(moead_files)
mean_hv_nsga, std_hv_nsga = calculate_statistics(nsga_files)


# Plot the results for both algorithms
plt.figure(figsize=(12, 6))

plt.plot(mean_hv_moead, label="MOEA/D Mean", color="red", linestyle="--", linewidth=2)
plt.fill_between(
    range(max_generation),
    mean_hv_moead - std_hv_moead,
    mean_hv_moead + std_hv_moead,
    color="lightcoral",
    alpha=0.3,
)

plt.plot(mean_hv_nsga, label="NSGA-II Mean", color="blue", linestyle="-", linewidth=2)
plt.fill_between(
    range(max_generation),
    mean_hv_nsga - std_hv_nsga,
    mean_hv_nsga + std_hv_nsga,
    color="lightblue",
    alpha=0.3,
)


plt.xlabel("Generation", fontsize=14)
plt.ylabel("Hyper Volume", fontsize=14)
plt.legend(fontsize=20)
plt.legend(loc="lower right")
plt.show()
