import math
from expression import integrate
from expression import differentiate
from expression import complex_number




# # print("Calculating the probability of finding an electron in a hydrogen atom within a certain range")

# # a = 5.29177 * 10**(-11)
# pi = math.pi
# pow = math.pow
# sqrt = math.sqrt
# exp = math.exp



lima = float(input("Enter the lower limit: "))
limb = float(input("Enter the upper limit: "))

expre = input("Please enter a valid python expression: ");

intObj = integrate(lima,limb,expre)
comp = complex_number(3**0.5,1)
print(comp.modulus())
# diffObj = differentiate('3*cos(2*x)',math.pi/4)

res = intObj.evaluateInt()
# resdiff = diffObj.evaluateDiff()

print(res)
# print(resdiff)