from operator import index
import random

N = 300
barrier_length = 1000
start_index = 0
end_index = 10


def is_all_zero(array):
    for i in array:
        if i != 0:
            return False
    return True


def initCoordinate(N):
    x = set()
    while x.__len__() < N:
        x.add(random.randint(1, barrier_length - 1))
    x = list(x)
    x.sort()
    return x


def save_array_to_txt(array, filename):
    with open(filename, "w") as file:
        for element in array:
            file.write(str(element) + "\n")


for index in range(start_index, end_index):
    filename = f"./dataset/{N}_{index}.txt"
    array = initCoordinate(N)
    save_array_to_txt(array, filename)
    print("Mảng đã được lưu vào tệp tin:", filename)

