import math
# import matplotlib.pyplot as plt
# import numpy as np

# # expression = input(" Please enter the function a function of x only: ")
# x = np.linspace(-10*math.pi,10*math.pi,num=1000)
# # exp = np.exp
# # sqrt = np.sqrt
# # power = np.power
# # pi = math.pi

# y = np.sin(x)
# angle = -math.pi/100

# y1 = y*math.cos(angle) + x*math.sin(angle)

# plt.xlabel("x-axis")
# plt.ylabel("y-axis")

# plt.plot(x,y,'r')
# plt.plot(x,y1,'b')

# plt.show()

'''from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
x = np.outer(np.linspace(-2, 2, 30), np.ones(30))
y = x.copy().T # transpose
z = np.sqrt(x**2 + y**2)
z1 = z*math.cos(math.pi/3 - np.sqrt(x**2+y**2) * math.sin(math.pi/3)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none')
ax.plot_surface(x, y, z1,cmap=cm.coolwarm, edgecolor='none')
ax.set_title('Surface plot')
plt.show()'''





