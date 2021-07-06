import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 500)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y, label='x=0,y=0')

xticks, yticks = ax.get_xticks(), ax.get_yticks()
xpos, ypos = 0, 0

for i,(x,y) in enumerate(zip(xticks, yticks)):
    if x == 0:
        xpos = i
    if y == 0:
        ypos = i
print(xpos, ypos)

x_min, x_max = ax.get_xlim()
xticks = [(tick - x_min)/(x_max - x_min) for tick in xticks]
y_min, y_max = ax.get_ylim()
yticks = [(tick - y_min)/(y_max - y_min) for tick in yticks]

print(xticks[xpos], yticks[ypos])
ax.legend(bbox_to_anchor=[xticks[xpos], yticks[ypos]], loc='center')
plt.show()