import matplotlib.pyplot as plt
import numpy as np

# Parameters for the circles and gradient
R_s = 2
r_u = 1
alpha = 0.8
n_layers = 100 
R_certain = R_s - r_u
primary_color = "green"

# Create a figure and axis
fig, ax = plt.subplots()

# Generate radii and alphas for gradient circles
radius = np.linspace(R_certain, R_s, n_layers)
alphas = np.exp(-np.linspace(-np.log(alpha), -np.log(0.01), n_layers))

# Draw gradient circles
for j in range(n_layers):
    circle = plt.Circle(
        (0, 0), radius[j], color=primary_color, fill=False, alpha=alphas[j]
    )
    ax.add_artist(circle)

# Draw inner filled circle (R_certain)
inner_circle = plt.Circle(
    (0, 0), R_certain, color=primary_color, alpha=alpha, ec="black"
)
ax.add_artist(inner_circle)

# Draw outer circle (R_s)
outer_circle = plt.Circle(
    (0, 0), R_s, edgecolor="gray", linestyle="--", linewidth=1.5, fill=False, alpha=0.3
)
ax.add_patch(outer_circle)

# Plot the center point 'S'
ax.plot(0, 0, "ko")
ax.text(0, 0, "S", fontsize=16, verticalalignment="bottom", horizontalalignment="right")

ax.annotate(
    "", xy=(R_s, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="<->", linewidth=1.5)
)
ax.text(R_s / 1.8, 0.1, "$R_S$", fontsize=14)

ax.annotate(
    "",
    xy=(0, R_certain),
    xytext=(0, R_s),
    arrowprops=dict(arrowstyle="<->", linewidth=1.5),
)
ax.text(0.1, (R_certain + R_s) / 2, "$R_U$", fontsize=14)

ax.set_aspect('equal')
ax.set_xlim(-R_s * 1.1, R_s * 1.1)
ax.set_ylim(-R_s * 1.1, R_s * 1.1)

# Hide axes
ax.axis('off')

plt.tight_layout()
plt.show()
