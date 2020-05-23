from graphics import *
import time
import numpy as np
import math
pi = math.pi
import random
r = random.randint
sqrt = math.sqrt

win = GraphWin('simulating a bouncing ball',1000,1000,autoflush=True)

width = 25
length = 900
g = 10

rectP1 = Point(50,650)
rectP2 = Point(rectP1.getX()+length,rectP1.getY()+width)

platform = Rectangle(rectP1,rectP2)
platform.setFill('cyan')
platform.draw(win)

centre = Point(rectP2.getX()/2,50)
ballRadius = 15



ball = Circle(centre,ballRadius)
ball.setFill('green')
ball.draw(win)

h = rectP1.getY()/2 - centre.getY()
vmax = sqrt(2*g*h)
u = 10

epochTime = time.time()

initTime = time.time()


while True:
    
    dt = (time.time() - initTime)
    # epoch = time.time() - epochTime
    # print('epoch == ',epoch)
    dy = u*dt+ 0.5*g*dt**2
    u = u + g*dt

    ball.move(0,dy)

    initTime = time.time()

    if(u >= vmax):
        
        ball.undraw()
        ball = Circle(centre,ballRadius)
        ball.setFill('green')
        ball.draw(win)
        u = 0

    
    win.update()




win.getMouse()
win.close()