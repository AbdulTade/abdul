from graphics import *
import time
import numpy as np
import math
pi = math.pi
import random
r = random.randint

sin = math.sin
cos = math.cos
colors = ['blue','cyan','magenta','yellow','red','green','orange','white']

win = GraphWin("Simulating a ball moving in a circular path",1000,1000,autoflush=True)

anchorPoint = Point(500,350)
ballRadius = 20
semiMinorAxis = 5 * ballRadius
semiMajorAxis = 15 * ballRadius

def Ellipse(windowObject,centre,semiMinorAxis,semiMajorAxis,color='black'):
    angles = np.linspace(-pi,pi,1000)
    x = centre.getX() + (semiMajorAxis * np.cos(angles))
    y = centre.getY() + (semiMinorAxis * np.sin(angles))
    # print(x,y)
    for k in range(1000):
            windowObject.plot(x[k],y[k],color)

    return [x,y]

ellipseColor = colors[r(0,len(colors)-1)]
ellip = Ellipse(win,anchorPoint,semiMinorAxis,semiMajorAxis,ellipseColor)


w = 1

initTime = time.time()

x = anchorPoint.getX() + semiMajorAxis
y = anchorPoint.getY() + semiMinorAxis
# print(x,y)

# ball = Circle(Point(x,y),ballRadius)
# ball.setFill('black')
# ball.draw(win)
wx = 10*w
wy = 10*w
# win.getMouse()

ballColor = colors[r(0,len(colors)-1)]

while True:

    dt = (time.time() - initTime)
    
    x = anchorPoint.getX() + semiMajorAxis * cos(wx*dt * (1/(2*pi)))
    y = anchorPoint.getY() + semiMinorAxis * sin(wy*dt * (1/(2*pi)))
    ball = Circle(Point(x,y),ballRadius)
    ball.setFill(ballColor)
    ball.draw(win)

    line = Line(anchorPoint,ball.getCenter())
    line.setWidth(2)
    line.setFill('black')
    line.draw(win)
    
    line.undraw()
    ball.undraw()
    win.update()
    