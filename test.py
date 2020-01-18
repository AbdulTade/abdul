import math
from expression import integrate
from expression import differentiate

intObj = integrate(2,3,'3*cos(2*x)')
diffObj = differentiate('3*cos(2*x)',math.pi/4)

res = intObj.evaluateInt()
resdiff = diffObj.evaluateDiff()


print(res)
print(resdiff)