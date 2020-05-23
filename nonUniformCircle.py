from graphics import *
import time
import numpy as np
import math
pi = math.pi
import random
r = random.randint

n = 1
sin = math.sin
cos = math.cos
colors = ['blue','cyan','magenta','yellow','red','green','orange','white']

win = GraphWin("Simulating a ball moving in a circular path",1000,1000,autoflush=True)

anchorPoint = Point(500,350)
ballRadius = 2
circleRadius = 15 * ballRadius


# circle = Circle(anchorPoint,circleRadius)
# circle.setOutline('black')
# circle.setWidth(3)
# circle.draw(win)

w = 100

initTime = time.time()

x = anchorPoint.getX() + circleRadius
y = anchorPoint.getY() 
# print(x,y)

# ball = Circle(Point(x,y),ballRadius)
# ball.setFill('black')
# ball.draw(win)

# win.getMouse()
wx = 1*w
wy = 1*w

ballColor = colors[r(0,len(colors)-1)]

while True:

    dt = (time.time() - initTime)
    
    x = anchorPoint.getX() + circleRadius * ((cos(wx*dt * (1/(2*pi))))**2)**(1/n)
    y = anchorPoint.getY() + circleRadius * ((sin(wy*dt * (1/(2*pi))))**2)**(1/n)

    x1 = anchorPoint.getX() - circleRadius * ((cos(wx*dt * (1/(2*pi))))**2)**(1/n)
    y1 = anchorPoint.getY() + circleRadius * ((sin(wy*dt * (1/(2*pi))))**2)**(1/n)

    x2 = anchorPoint.getX() + circleRadius * ((cos(wy*dt * (1/(2*pi))))**2)**(1/n)
    y2 = anchorPoint.getY() - circleRadius * ((sin(wy*dt * (1/(2*pi))))**2)**(1/n)
    
    x3 = anchorPoint.getX() - circleRadius * ((sin(wy*dt * (1/(2*pi))))**2)**(1/n)
    y3 = anchorPoint.getY() - circleRadius * ((sin(wy*dt * (1/(2*pi))))**2)**(1/n)
    
    ball = Circle(Point(x,y),ballRadius)
    ball.setFill(ballColor)
    ball.draw(win)
    win.plot(x,y,color='blue')
    win.plot(x1,y1,color='blue')
    win.plot(x2,y2,color='blue')
    win.plot(x3,y3,color='blue')

    line = Line(anchorPoint,ball.getCenter())
    line.setWidth(2)
    line.setFill('black')
    line.draw(win)
    
    line.undraw()
    ball.undraw()
    win.update()