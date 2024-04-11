# Just a simple script, standalone because matplotlib is not really a dependency of the assignment

import matplotlib.pyplot as plt
import json

with open("accuracies.json", "r") as f:
    data = json.load(f)
x = data["epochs"]
y = data["accuracies"]

x, y = zip(*sorted(zip(x, y)))

fig = plt.figure()
plt.plot(x, y)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Epoch fp32")
fig.savefig("accuracies_fp32.png")