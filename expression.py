import math
# expression = '2*x**2+x-3*e(-x)'
# x = 3
# e = math.exp
# arr = list(expression)

# res = expression.split('+')

# print(res)

# print(eval(expression))

e  =  math.exp
log = math.log10
cos = math.cos
sin = math.sin
tan = math.tan
pi = math.pi
sqrt = math.sqrt

class differentiate:

    def __init__(self,expression,xval):

        self.x = xval
        self.expression = expression

    def evaluateDiff(self):

        delta = 10**(-10)
        x = self.x
        funcValInit = eval(self.expression)
        x = self.x+delta
        funcValFinal = eval(self.expression)

        diffVal = (funcValFinal - funcValInit)/delta
        return diffVal


                


class integrate:

    def __init__(self,lima,limb,equation):

        self.a = lima
        self.b = limb 
        self.equation = equation
        self.n = 100000

   

    def evaluateInt(self):

        
        interval = (self.b - self.a)/self.n
        i = 0
        Areasum = 0
        while(i < self.n):

            x = self.a + interval*i
            Areasum += interval * eval(self.equation)

            i+=1

        return Areasum


class complex_number:

    def __init__(self,Re,Im):

        self.Re = Re
        self.Im = Im

    def modulus(self):

        return sqrt(self.Re**2 + self.Im**2)

    def argz(self):

        return math.atan(self.Im/self.Re)

    def conjugate(self):
        
        print('({},{})'.format(self.Re,self.Im))
        return complex_number(self.Re,-self.Im)

    def polar(self):

        val = tuple(self.modulus(),self.argz())
        return val



    




