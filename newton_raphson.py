import numpy as np 
import math
from expression import differentiate
sin = math.sin
equa = "x - sin(x) "

def swap(x,y):
    temp = x
    x = y 
    y = temp
    
def fprime(expression,value):
    diff = differentiate(expression,value)
    res = diff.evaluateDiff()
    return res

xval = 0.1

def newRaph(expression_equation,numOfiterations):
    i = 0
    x = xval
    
    while(i < numOfiterations):
         xiter = xval - fprime(expression_equation,xval)/eval(expression_equation)
         swap(xval,xiter)
         x = xiter
         i += 1
    return xiter

testValue = newRaph(equa,20000)
print(testValue)


