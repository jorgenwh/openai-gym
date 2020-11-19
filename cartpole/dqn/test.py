import numpy as np


tensor = np.arange(1,13).reshape(6,2)
rewards = np.zeros((6, 1))

mask = np.zeros((6,1))
for i in range(6):
    mask[i, 0] = np.random.randint(0,2)

mask = mask.astype("int")

print(tensor)
print()
print(rewards)
print()
print(mask)
print()

print(tensor[:,mask])
