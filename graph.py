import math
import sys
import time
import random
from graphics import *
import numpy as np
import math
from QsuiteGui import norm



win = GraphWin('Pendulum',height=1000,width=1000,autoflush=True)
win.setBackground('white')
rectwidth = 20
rectlength = 500

p0 = np.array((50,50))
p1 = np.array((50+rectlength,50+rectwidth))

rect = Rectangle(Point(50,50),Point(50+rectlength,50+rectwidth))
rect.setFill('blue')
rect.draw(win)

midPoint = np.array(( (p0[0] + p1[0])/2, (p0[1]+p1[1]+rectwidth)/2 ))

centre = np.array((300,300))

v1 = p1 - midPoint
v2 = centre - midPoint
angle = math.acos( np.dot(v1,v2)/(norm(v1,len(v1)) * norm(v2,len(v2))) )
startTime = time.time()
g = 9.81
T = 2*math.pi*math.sqrt(norm(v2,len(v2))/g)
w = 2*math.pi/T

angleState = False
line = Line(Point(midPoint[0],midPoint[1]),Point(centre[0],centre[1]))
circ = Circle(Point(centre[0],centre[1]),5)
circ.setFill('blue')
circ.draw(win)
line.setFill('blue')
# line.setWidth(20)
line.draw(win)

text = Text(Point(900,50),'Test')
text.setFill('green')
text.setSize(10)
text.setStyle('italic')
text.setFace('arial')
text.draw(win)

epochTime = time.time()
Oscillations = 0

while(True):
    
    elapsedTime = time.time() - startTime
    angle =  angle + w * elapsedTime
    startTime = time.time()

    line.undraw()
    circ.undraw()

    if(angle >= math.pi):
        angleState = True

    while(angleState):

        # line.undraw()
        # circ.undraw()
        elapsedTime = time.time() - startTime
        angle =  angle - w * elapsedTime
        startTime = time.time()

        # line.undraw()
        # circ.undraw()

        dx = norm(v2,len(v2)) * math.cos(angle)
        dy = norm(v2,len(v2)) * math.sin(angle)
        x = centre[0] + dx
        y = centre[1] + dy

        # time.sleep(0.05)

        line = Line(Point(midPoint[0],midPoint[1]),Point(x,y))
        circ = Circle(Point(x,y),20)
        circ.setFill('blue')
        circ.draw(win)
        win.plot(x,y,'blue')
        line.setFill('blue')
        line.setWidth(5)
        line.draw(win)
        text.setText('Duration: {}s\nno Oscillations: {}\nPeriod: {}'.format(time.time()-epochTime,Oscillations,T))
        line.undraw()
        circ.undraw()

        win.update()
        if(angle <= 0):
            angleState = False
            Oscillations += 1
            break
    
    dx = norm(v2,len(v2)) * math.cos(angle)
    dy = norm(v2,len(v2)) * math.sin(angle)
    x = centre[0] + dx
    y = centre[1] + dy

    # time.sleep(0.05)

    line = Line(Point(midPoint[0],midPoint[1]),Point(x,y))
    circ = Circle(Point(x,y),20)
    circ.setFill('blue')
    circ.draw(win)
    win.plot(x,y,'red')
    line.setFill('blue')
    line.setWidth(5)
    text.setText('Duration: {}s\nno Oscillations: {}\nPeriod: {}'.format(time.time()-epochTime,Oscillations,T))
    line.draw(win)


    win.update()
