import pandas as pd
import matplotlib.pyplot as plt

file_nsga = "result/nsga_time.csv"
file_moead = "result/moead_time.csv"

data_nsga = pd.read_csv(file_nsga, header=None)
data_moead = pd.read_csv(file_moead, header=None)

columns = ["pop_size", "max_gen", "dataset", "run", "runtime"]
data_nsga.columns = columns
data_moead.columns = columns

data_nsga = data_nsga[["dataset", "runtime"]]
data_moead = data_moead[["dataset", "runtime"]]

data_nsga["prefix"] = data_nsga["dataset"].str.split("_").str[0]
data_moead["prefix"] = data_moead["dataset"].str.split("_").str[0]

avg_nsga = (
    data_nsga.groupby("prefix")["runtime"].mean().reset_index(name="nsga_avg_runtime")
)
avg_moead = (
    data_moead.groupby("prefix")["runtime"].mean().reset_index(name="moead_avg_runtime")
)

combined_data = pd.merge(avg_nsga, avg_moead, on="prefix")

plt.figure(figsize=(10, 6))
plt.plot(
    combined_data["prefix"],
    combined_data["nsga_avg_runtime"],
    marker="o",
    color="blue",
    label="NSGA-II",
    markersize=9,
)
plt.plot(
    combined_data["prefix"],
    combined_data["moead_avg_runtime"],
    marker="D",
    color="red",
    label="MOEA/D",
    markersize=9,
)

plt.xlabel("Number of sensors", fontsize=14)
plt.ylabel("Mean run time (s)", fontsize=14)
plt.legend(fontsize=14)
plt.grid()
plt.show()
