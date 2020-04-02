from graphics import *
import numpy as np
import math

win = GraphWin("Vector field of v(x,y) = yi + xj ", 1100, 1100)
win.setBackground('white')
x = np.linspace(10,900,300)
y = np.linspace(10,900,300)

c = 100/(np.sqrt(x**2 +  y**2)) 
i = 0
j = 0
point_lis = []
Pt = []

while(i < len(x)):
    while(j < len(y)):
        point_lis.append((y[i]**0.8,x[j]**0.8))
        Pt.append(Point(y[i],x[j]))

        j += 1
    i += 1

endPoint = []
k = 0
while(k < len(x)):
    xPt = point_lis[k][0] + x[k]
    yPt = point_lis[k][1] + y[k]
    endPoint.append(Point(xPt,yPt))
    k += 1

z = 0
while(z < len(x)):
    line = Line(Pt[z],endPoint[z])
    line.draw(win)
    z += 1

win.getMouse()
win.close()



