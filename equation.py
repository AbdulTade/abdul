from graphics import *
import numpy as np

win = GraphWin("x**n + y**n = r**n",1000,1000)

x = np.linspace(0,900,900)
n = 5
r = 200
y = (r**n - x**n)**(1/n)

PointsPos = []
PointsNeg = []

xlen = len(x)
for i in range (0, xlen):

    PointsPos.append(Point(x[i],y[i]))
    PointsNeg.append(Point(x[i],-y[i]))
    

for i in range(0,xlen):

    Line(PointsPos[i],Point(x[i]+1,y[i]+1)).draw(win)
    Line(PointsNeg[i],Point(x[i]+1,-y[i]+1)).draw(win)

win.getMouse()
win.close()




