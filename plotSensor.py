import matplotlib.pyplot as plt
import numpy as np
fileData = "./dataset/100_1.txt"
# Mảng hoành độ tâm vòng tròn
x_pos = np.loadtxt(fileData, dtype=int)
N = len(x_pos)
# Mảng bán kính vòng tròn
radii = np.array(
    [14.0, 0, 0, 0, 14.0, 0, 0, 0, 6.0, 12.5, 12.5, 3.5, 0, 20.5, 0, 0, 0, 0, 0, 20.5, 0, 4.0, 0, 3.5, 0, 6.0, 0, 0, 0, 0, 23.0, 0, 0, 0, 23.0, 0, 13.5, 0, 13.5, 0, 0, 2.5, 0, 12.0, 0, 14.0, 0, 0, 14.5, 0, 0, 0, 0, 0, 0, 26.5, 0, 0, 0, 26.5, 0, 0, 21.0, 0, 0, 0, 21.0,
     0, 0, 14.0, 0, 37.0, 0, 0, 0, 0, 0, 0, 0, 58.0, 0, 0, 58.0, 0, 0, 0, 16.5, 0, 0, 0, 0, 24.0, 0, 24.0, 15.0, 0, 36, 0, 0, 0]
)
fig, ax = plt.subplots()

# Vẽ các vòng tròn
for i in range(N):
    circle = plt.Circle((x_pos[i], 0), radii[i], color='blue', alpha=0.5)
    ax.add_patch(circle)
    if radii[i] > 0:
        ax.scatter(x_pos[i], 0, marker='o', color='red',
                   s=10)  # Chấm đỏ có kích thước 10
    else:
        ax.scatter(x_pos[i], 0, marker='x', color='black',
                   s=10)  # Chấm đen có kích thước 10
# Thiết lập trục
ax.set_xlim(0, 1000)  # Thiết lập giới hạn trục X
ax.set_ylim(-radii.max(), radii.max())  # Thiết lập giới hạn trục Y
ax.set_aspect('equal')  # Giữ tỷ lệ khung hình

# Thêm nhãn trục
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Hiển thị đồ thị
plt.show()
