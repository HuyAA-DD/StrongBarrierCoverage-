import pandas as pd
import matplotlib.pyplot as plt

dataset = "100"
algo = "nspso"
#algo = "nsga"
subset = 0
run = 0
filename = f"./result/f/{algo}/{algo}_{dataset}_{subset}_{run}.csv"
df = pd.read_csv(filename, sep=" ", header=None, skiprows=1)
df.columns = ["f1", "f2"]

pop_size = 32

# Số thế hệ cụ thể mà bạn muốn biểu diễn
selected_generations = [1, 10, 50, 100, 500, 1000, 10000]

colors = ["y", "c", "m", "g", "b", "k", "r"]
markers = ["<", "D", ">", "^", "s", "v", "o"]

plt.figure(figsize=(10, 9))

for i, gen in enumerate(selected_generations):
    if gen * pop_size <= len(df):
        gen_data = df.iloc[(gen - 1) * pop_size : gen * pop_size]
        plt.scatter(
            gen_data["f1"],
            gen_data["f2"],
            label=f"Gen {gen}",
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            s=70,
        )
        if gen == selected_generations[-1]:
            gen_data = gen_data.sort_values(by="f1")
            plt.plot(
                gen_data["f1"],
                gen_data["f2"],
                "--",
                color=colors[i % len(colors)],
                linewidth=2.5,
                # label=f"Gen {gen} (connected)",
            )


plt.xlabel("Number of active sensors", fontsize=14)
plt.ylabel("Total energy consumption", fontsize=14)
plt.legend(fontsize=19)
plt.grid(True)
plt.tight_layout()
plt.show()
