import numpy as np
import matplotlib.pyplot as plt

fileData = "./dataset/100_1.txt"
x_corr = np.loadtxt(fileData, dtype=int)
N = len(x_corr)
N = 2

k = 2
alpha = 0.7
n_layers = 100
primary_color = "forestgreen"

radii = np.array([])
fig, ax = plt.subplots(figsize=(10, 6))

for i in range(N):
    if radii[i] > 0:
        r_u = radii[i]
        R_s = k * r_u
        R_certain = R_s - r_u
        radius = np.linspace(R_certain, R_s, n_layers)
        alphas = np.exp(-np.linspace(-np.log(alpha), -np.log(0.01), n_layers))

        for j in range(n_layers):
            circle = plt.Circle(
                (x_corr[i], 0),
                radius[j],
                color=primary_color,
                fill=False,
                alpha=alphas[j],
            )
            ax.add_artist(circle)

        inner_circle = plt.Circle(
            (x_corr[i], 0), R_s - r_u, color=primary_color, alpha=alpha, ec="black"
        )
        ax.add_artist(inner_circle)

        outer_circle = plt.Circle(
            (x_corr[i], 0),
            R_s,
            edgecolor="gray",
            linestyle="--",
            linewidth=1.5,
            fill=False,
            alpha=0.3,
        )
        ax.add_patch(outer_circle)

        ax.scatter(x_corr[i], 0, marker="o", color="red", s=10)
    else:
        ax.scatter(x_corr[i], 0, marker="x", color="black", s=10)

ax.set_xlim(0, 1000)
ax.set_ylim(-R_s - 1, R_s + 1)
ax.set_aspect("equal")
plt.axis("off")
plt.tight_layout()
plt.show()
