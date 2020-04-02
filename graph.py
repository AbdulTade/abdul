# from graphics import * 
# import numpy as np
import math
import sys
import datetime
import random
import pygame
from pygame.locals import *


height = 1000
BLACK =(0,0,0)
width = 1000
allowance = 250
length = 350
acceleration = 10
size = random.randint(1,10)
T = 2*math.pi*math.sqrt(length / acceleration)
angularFrequency = 2*math.pi/T
FPS = 60

fpsclock = pygame.time.Clock()

rectLen = 500
rectWidth = 15
pygame.init()
surf = pygame.display.set_mode((1000,1000))
surf.fill((255,255,255))
pygame.display.set_caption(' Simple Pendulum ')

pygame.draw.rect(surf,(0,0,0),(allowance,15,rectLen,rectWidth))
midPtX = allowance + 0.5*(rectLen)
midPtY = 15 + rectWidth
midPt = (midPtX,midPtY)

destPoint = (int(midPtX) ,int(midPtY + length))
pygame.draw.line(surf,BLACK,midPt,destPoint,2)
circ = pygame.draw.circle(surf,BLACK,destPoint,10)

max_angle = math.pi/size
time_unit = T*100/size
velocity = 2*math.pi*length/T
print(velocity)

def motion(amplitude_angle,num_divisions):
     angle_unit = amplitude_angle/num_divisions
     while(amplitude_angle <= max_angle):
         amplitude_angle += angle_unit
         break
     while(amplitude_angle > max_angle):
        amplitude_angle -= angle_unit
        break

def swing(angle):
    destPoint = (int(midPtX + length*math.cos(angle)) ,int(midPtY + length*math.sin(angle)))
    pygame.draw.line(surf,BLACK,midPt,destPoint,2)
    circ = pygame.draw.circle(surf,BLACK,destPoint,10)
    pygame.time.wait(int(time_unit))



while True:
    swing(max_angle)
    motion(max_angle,100)
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    pygame.display.update()
    fpsclock.tick(FPS)




# def today(dateToday):
#     return tuple([dateToday.hour,dateToday.minute,dateToday.second,dateToday.microsecond])

# start = today(datetime.datetime.now())

# def Interval(now,next):
#     j = 0
#     time_lis = []
#     while(j < 4):
#           time_lis.append(next[j] - now[j])
#           j+=1
#     interval = time_lis[0]*3600 + time_lis[1]*60 + time_lis[2] + time_lis[3]
#     return interval



# def main():

#     while(True):

#         win = GraphWin("My Window",width,height)
#         win.setBackground('blue')
#         now = today(datetime.datetime.now())
#         txt1 = Text(Point(900,10),"Time {}:{}:{}".format(now[0],now[1],now[2]))
#         destPoint = Point(width/2 ,height/10)
        

#         pt1 = Point(allowance,height/10)
#         pt2 = Point(width-allowance,height/10)
#         midpt = Point(width/2,height/10)
#         baseline = Line(pt1,pt2)    
#         angle = angularFrequency*Interval(start,now)
#         destPoint.move(length*math.cos(angle),length*math.sin(angle))
        
#         axisline = Line(midpt,destPoint)
#         poly = Polygon(Point(allowance,height/10 - 15),pt1,pt2,Point(width-allowance,height/10 - 15))
#         circ = Circle(destPoint,10)
#         txt2 = Text(Point(900,50),"x = {} y = {} ".format(width/2 + length*math.cos(angle - math.pi),height/10 + length*math.sin(angle - math.pi)))
#         circ.setFill("red")
#         poly.setFill("black")
        
#         axisline.draw(win)
#         circ.draw(win)
#         destPoint.draw(win)
#         baseline.draw(win)
#         poly.draw(win)
#         txt1.draw(win)
#         txt2.draw(win)

    
#         win.update()
#         # axisline.undraw()
#         # axisline.undraw()
#         win.getMouse()
#         win.close()

