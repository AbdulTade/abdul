from graphics import *
import time
import math
import  numpy as np
import random
r = random.randint

win = GraphWin('Simulating an electron deflected in the magnetic field',1000,1000,autoflush=True)
length = 450
width = 10
d = 200

radius = 10

E = 100 # in V/m(Volt per meter)
# B = 0.01 # in T(tesla)

m = 9.1 * 10**(-31)
q = 1.6 * 10**(-27)

# v = E/B

def TextFormatter(point,text):
    text = Text(point,text)
    text.setFill('black')
    text.setSize(10)
    text.setStyle('italic')
    text.draw(win)

ux = 5

electric1P1 = Point(300,200)
electric1P2 = Point(electric1P1.getX()+length,electric1P1.getY()+width)

electric2P1 = Point(electric1P1.getX(),electric1P1.getY()+d+width)
electric2P2 = Point(electric1P1.getX()+length,electric1P1.getY()+(2 * width)+d)

centy = (electric1P1.getY() + electric2P2.getY())/2
centre = Point(10,centy)

TextFormatter(Point(electric1P1.getX() + (0.5*length),electric1P1.getY()-50),'Electric plates')

electricPlate1 = Rectangle(electric1P1,electric1P2)
electricPlate1.setFill('grey')
electricPlate2 = Rectangle(electric2P1,electric2P2)
electricPlate2.setFill('grey')

electricPlate2.draw(win)
electricPlate1.draw(win)

ball = Circle(centre,radius)
ball.setFill('black')
ball.draw(win)

timeToReverse = length/ux
print(timeToReverse)
initTime = time.time()
uy = 0
ay = ((q*E)/m) * 0.001
dy = 0
y = (electric1P1.getY() + electric2P2.getY())/2
print(ay)
while True:
    dt = (time.time() - initTime)
    # print(dt)
    dx =  dt * ux * 0.1
    uy = uy + ay*dt
    # print('uy = ',uy)
    if(ball.getCenter().getX() >= electric1P1.getX()):
        dy =  uy * dt * 0.000005
    # ux = ((q*E)/m) * dt * 0.1
    # print(dx,dy)
    ball.move(dx,dy)
    win.update()

win.getMouse()
win.close()