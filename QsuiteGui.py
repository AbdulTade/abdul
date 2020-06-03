import math
import random
from numpy import linalg as LA
import matplotlib.pyplot as plt
import unicodedata

# expression = '2*x**2+x-3*e(-x)'
# x = 3
# e = math.exp
# arr = list(expression)iskit

# res = expression.split('+')

# print(res)

# print(eval(expression))

exp  =  math.exp
log = math.log10
cos = math.cos
sin = math.sin
tan = math.tan
asin = math.asin
acos = math.acos # Create refrences to usual math functions so that it can be called with eval
atan = math.atan
pi = math.pi
sqrt = math.sqrt
erf = math.erf
erfc = math.erfc
tanh = math.tanh
cosh = math.cosh
sinh = math.sinh
atanh = math.atanh
acosh = math.acosh
asinh = math.asinh

def round(num):
    wholeNum = int(str(num).split('.')[0])
    rem = num - wholeNum
    if(rem*2 < 1):
        return wholeNum
    elif(rem*2 >= 1):
        return wholeNum+1

def ndecimalPlaces(num,n):
    whole = int(str(num).split('.')[0])
    rem = num - whole
    dp = round(10**n * rem) / 10**(n)
    return whole+dp

def negativeDetector(num):
            if(num < 0):
                num = -num
                return ' - {}'.format(num)
            if(num == 0):
                return '+ 0'
            if(num > 0):
                return ' + {}'.format(num)


class differentiate:

    def __init__(self,expression,xval):

        self.x = xval
        self.expression = expression # Initialize required values of expression and value whose differrential is to be found

    def evaluateDiff(self): # function to evaluate integral 

        delta = 10**(-10) # xvalue difference
        x = self.x # initial xvalue
        funcValInit = eval(self.expression) # initial function value
        x = self.x+delta # final xvalue
        funcValFinal = eval(self.expression) # final function value

        diffVal = (funcValFinal - funcValInit)/delta
        return diffVal


                


class integrate:

    def __init__(self,lima,limb,equation):

        self.a = lima # intialize data values
        self.b = limb 
        self.equation = equation
        self.n = 100000

   

    def evaluateInt(self):
        
        interval = (self.b - self.a)/self.n
        i = 0
        Areasum = 0
        while(i < self.n):

            x = self.a + interval*i
            Areasum += interval * eval(self.equation) # strip size for integral 100 trillionth

            i+=1

        return Areasum


class complex_number:

    def __init__(self,Re,Im):

        self.Re = Re
        self.Im = Im

    def __repr__(self):

        return 'z = {}{}*j'.format(ndecimalPlaces(self.Re,3),negativeDetector(self.Im))

    def __add__(self,complexObject):
        return complex_number((self.Re + complexObject.Re),(self.Im + complexObject.Im))
    
    def __sub__(self,complexObject):
        return complex_number((self.Re - complexObject.Re),(self.Im - complexObject.Im))

    def __mul__(self,complexObject):
        xRe = self.Re*complexObject.Re - self.Im*complexObject.Im
        yIm = self.Re*complexObject.Im + self.Im*complexObject.Re
        return complex_number(xRe,yIm)

    def __truediv__(self,complexObject):
        rsquared = 1/complexObject.modulus()**2 
        xRe = rsquared * (self.Re*complexObject.Re + self.Im*complexObject.Im)
        yIm = rsquared * (self.Im*complexObject.Re - self.Re*complexObject.Im)
        return complex_number(xRe,yIm)

    def __lt__(self,complexObject):
        return (self.Im < complexObject.Im) and (self.Re < complexObject.Re)

    def __gt__(self,complexObject):
        return (self.Im > complexObject.Im) and (self.Re > complexObject.Re)

    def __eq__(self,complexObject):
        return (self.Im == complexObject.Im) and (self.Re == complexObject.Re)

    def __le__(self,complexObject):
        return (self.Im <= complexObject.Im) and (self.Re <= complexObject.Re)

    def __ge__(self,complexObject):
        return (self.Im >= complexObject.Im) and (self.Re >= complexObject.Re)

    def comp_scale(self,scalar):
        self.Re = self.Re*scalar
        self.Im = self.Im*scalar

    def modulus(self):
        # print('{}{}'.format(unicodedata.lookup('SQUARE ROOT'),rootSquare))
        return math.sqrt(self.Re**2 + self.Im**2)

    def argz(self):

        return math.atan(self.Im/self.Re)

    def conjugate(self):

        return complex_number(self.Re,-self.Im)

    def polar(self):

        val = (self.modulus(),self.argz())
        return val

    def __pow__(self,n):

        (modulus,argz) = self.polar()
        modulus = modulus**n
        argz = n*argz
        return complex_number(modulus*math.cos(argz),modulus*math.sin(argz))
        
    def inverse_complex_power(self,n):
        (modulus,argz) = self.polar()
        arg_lis = []
        modulus = modulus**(1/n)
        for k in range(n):
            arg_lis.append(2*math.pi*k*(1/n))
        return [complex_number(modulus*cos(argz/n + arg_lis[k]),modulus*sin(argz/n + arg_lis[k])) for k in range(n)]
        
    def nPolygon(self,n):
        if(n < 3):
            print("Cannot form a closed 2d figure with only {} point/s ".format(n))
            return None
        mod = self.modulus()
        circ = plt.Circle((0,0),1.1,color='y',clip_on=False)
        fig,ax = plt.subplots()
        ax.add_artist(circ)
        plt.title('No. sides = {}'.format(n))
        comp_res = self.inverse_complex_power(n)
        x1 = [comp_res[i].Re for i in range(n)]
        y1 = [comp_res[i].Im for i in range(n)]
        plt.scatter(x1,y1,s=100)
        for j in range(len(comp_res)):
            p = plt.Line2D([x1[j],x1[j-1]],[y1[j],y1[j-1]],linewidth=2,alpha=0.8)
            ax.add_artist(p)
        plt.show()
        return None

    def MandelBrotPlot(self,cvalue):
        zlis = []
        for i in range(100):
            zlis.append(complex_number(self.Re,self.Im))
            self.Re = self.Re**2 - self.Im**2 + cvalue.Re
            self.Im = 2*self.Re*self.Im + cvalue.Im
        xvals = np.array([zlis[k].Re for k in range(100)])
        yvals = np.array([zlis[j].Im for j in range(100)])
        plt.title('Complex Number z')
        plt.xlabel('Im(z)')
        plt.ylabel('Re(z)')
        plt.plot(xvals,yvals,color='r')
        plt.show()
        
    def cexp(self):
        r = exp(-self.Im)
        self.Im = r*sin(self.Re)
        self.Re = r*cos(self.Re)

    # def rotate(self,angle):

        



class complex_function:

    def __init__(self,utRe,vtIm):

        self.u = utRe
        self.v = vtIm

    def __repr__(self):
        return '{} + {}*i'.format(self.u,self.v)

    def complex_diff(self,tvalue):
        
        xdiff = differentiate(self.u,tvalue).evaluateDiff()
        ydiff = differentiate(self.v,tvalue).evaluateDiff()
        return complex_number(xdiff,ydiff)

    def complex_int(self,ta,tb):
        xInt = integrate(ta,tb,self.u).evaluateInt()
        yInt = integrate(ta,tb,self.v).evaluateInt()
        return complex_number(xInt,yInt)


        
        


import numpy as np 
class polynomial:

    def __init__(self,coefficient_vector):

        self.cvector = np.array(coefficient_vector)
        self.length = len(coefficient_vector)

    def  __repr__(self):

        
        j = 0
        expression = ''
        while(j < self.len):
            if(j == 0):
                expression += str(self.cvector[j]) + ' + '
            if(self.cvector[j] == 0):
                j += 1
                continue
            expression += " {}*x**{} + ".format(self.cvector[j],j)
            j += 1
        return expression

    def poly_eval(self,xval):
        poly_val = 0
        for j in range(self.length):
            poly_val += self.cvector[j]*xval**j
        return poly_val

    def poly_add(self,second_coeffient_vector):
        i = 0
        lis = []

        def lenChecker(len1,len2):
            if(len1 <= len2):
                return len1
            if(len2 <= len1):
                return len2

        def joiner(vec1,vec2,start_index):
           if(len(vec1) >= len(vec2)):
            return vec1[start_index:len(vec1)]
           if(len(vec2) >= len(vec1)):
            return vec2[start_index:len(vec2)]


        while(i < lenChecker(self.length,len(second_coeffient_vector))):
             lis.append(self.cvector[i] + second_coeffient_vector[i])
             i += 1
             lis = joiner(self.cvector,second_coeffient_vector,i)
        return polynomial(lis)

    def poly_subtract(self,second_coeffient_vector):
        k = 0
        sub_lis = []

        def lenChecker(len1,len2):
            if(len1 <= len2):
                return len1
            if(len2 <= len1):
                return len2
        trueLen = lenChecker(self.length,len(second_coeffient_vector))
        while(k < trueLen):
            sub_lis[k] = self.cvector[k] - second_coeffient_vector[k]
            k += 1
        return polynomial(sub_lis)

    def poly_multiplication(self,second_coeffient_vector):
        n = 0
        m = 0
        mult_lis = []
        term = []
        len2 = len(second_coeffient_vector)

        while(m < len2):
            term.append(self.cvector*second_coeffient_vector[m])
            mult_lis.append(term)
            term = []
            m += 1
        j = 0
        print(mult_lis)
        # while(j < len(mult_lis))
        return polynomial(mult_lis)

    def poly_diff(self):

        def leftshift(vec,length):
            j = 1
            while(j < length ):
                vec[j-1] = vec[j]
                j += 1
            vec[j-1] = 0
            return vec

        diff_lis = []
        k = 0
        while( k < self.length):
            diff_lis.append(k * self.cvector[k])
            k += 1
        diff_lis = leftshift(diff_lis,len(diff_lis))
        return polynomial(diff_lis)

    def poly_int(self):

        def rightshift(vec,length):
            a = 0
            while(a < length-1):
                vec[1-a] = vec[a]
                a += 1
            vec[0] = 3
            return vec

        int_lis = []
        n = 0
        while(n < self.length):
            int_lis.append(1/(n+1) * self.cvector[n])
            n += 1
        
        int_lis = rightshift(int_lis,len(int_lis))
        #print(int_lis)
        return polynomial(int_lis)




def norm(vector,arrlen):
     k = 0
     squared_norm = 0
     while(k < arrlen):
         squared_norm += vector[k]**2
         k+= 1
     return math.sqrt(squared_norm)

def proj(vec1,vec2):
  dot = np.dot(vec1,vec2)
  norm2squared = norm(vec2,len(vec2))**2
  k = dot/norm2squared
  return k*np.array(vec2)

def orthonormalize(vecArray):
    i = 1
    j = 0
    vectorlen = len(vecArray[0])
    orthonormal_list = []
    orthonormal_list.append(vecArray[0])
    while(i < vectorlen):
        while( j < vectorlen-1):
            orthitem = vecArray[i] - proj(vecArray[i],orthonormal_list[j])
            vecArray[i] = orthitem
            j += 1
            orthonormal_list.append(orthitem)
        i += 1
    i = 0
    orthonormal_list = np.array(orthonormal_list)
    print(orthonormal_list)
    while(i < len(orthonormal_list)):
        k = 1/norm(orthonormal_list[i],len(orthonormal_list))
        orthonormal_list[i] = k*orthonormal_list[i]
        i += 1
    
    return orthonormal_list


class statistics:

    def __init__(self,vectorArray):
        self.vec = np.array(vectorArray)
        self.len = len(vectorArray)

    def mean(self):
        return np.mean(self.vec)

    def median(self):
        return np.median(self.vec)

    def mode(self):
        return np.mode(self.vec)

    def std(self):
        return np.std(self.vec)

    def zscores(self):
        return 1/(self.std()) * (self.vec - np.ones(len(self.vec))*self.mean())



def collatz(num):

    collatz_list = []
    while(num > 1):
         if(num == 1):
              collatz_list.append(1)
              print(collatz_list)
              break
         if(num%2):
              collatz_list.append(num)
              num = 3*num + 1
              
         else:
              collatz_list.append(num)
              num = num//2
    xvals = np.linspace(0,len(collatz_list)-1,len(collatz_list))
    return [xvals,np.array(collatz_list)]

def collatz_plot(array):

    x = array[0]
    y = array[1]
    plt.scatter(x,y,s=20)
    plt.show()



r = random.randint
callable(r)

exp  =  math.exp
log = math.log10
cos = math.cos
sin = math.sin
tan = math.tan
pi = math.pi
sqrt = math.sqrt
erf = math.erf
erfc = math.erfc
tanh = math.tanh
cosh = math.cosh
sinh = math.sinh
atanh = math.atanh
acosh = math.acosh
asinh = math.asinh 

def randomGenerator(num):
    expression = ''
    expreJson = {
        0:'{}*exp({}*x)'.format(r(-100,100),r(-100,100)),
        1:'{}*log({}*x)'.format(r(-100,100),r(-100,100)),
        2:'{}*cos({}*x)'.format(r(-100,100),r(-100,100)),
        3:'{}*sin({}*x)'.format(r(-100,100),r(-100,100)),
        4:'{}*tan({}*x)'.format(r(-100,100),r(-100,100)),
        5:'pi*{}'.format(r(-100,100)),
        6:'{}*sqrt(x)'.format(r(-100,100)),
        7:'{}*x**{}'.format(r(-100,100),r(0,10)),
        8:'{}*erf({}*x)'.format(r(-100,100),r(-100,100)),
        9:'{}*erfc({}*x)'.format(r(-100,100),r(-100,100)),
        10:'{}*tanh({}*x)'.format(r(-100,100),r(-100,100)),
        11:'{}*sinh({}*x)'.format(r(-100,100),r(-100,100)),
        12:'{}*cosh({}*x'.format(r(-100,100),r(-100,100)),
        13:'{}*atanh({}*x)'.format(r(-100,100),r(-100,100)),
        14:'{}*asinh({}*x)'.format(r(-100,100),r(-100,100)),
        15:'{}*acosh({}*x'.format(r(-100,100),r(-100,100))
    }
    k = 0
    while(k < num):
        if(k == num):
            expression += expreJson[r(0,15)]
        expression += expreJson[r(0,15)] + ' + '
        k += 1
    print(expression)
    expression += '0'
    return expression


def Kofunction(cofPair,k):
    '''NB: The cof-pair should be a list of the function strings of length 2'''
    j = 0
    z = 0
    # plus_minus_pattern = []
    while(j < k):
        x = j+1
        z += (-1)**eval(cofPair[0]) * eval(cofPair[1])
        j += 1

    return [[k,z],cofPair]

# def complementary_Kofunction(cofPair,k):
#     m = 0
#     z = 0
#     while(m < k):
#         x = k-m
#         z += (-1)**eval(cofPair[0]) * eval(cofPair[1])
#         m += 1

    # return [[k,z],cofPair]

def Diracdelta(x,offset=0):
    if(x == offset):
        return 1000000000000000
    else:
        return 0
    


class vectorFunction:

    def __init__(self,xt,yt,zt):
       self.xt = xt
       self.yt = yt
       self.zt = zt
    
    def __repr__(self):
        expression = '({})i + ({})j + ({})k'.format(self.xt,self.yt,self.zt)
        return expression
    
    def vecDiff(self,param):
        x = param
        xdiff = differentiate(self.xt,x).evaluateDiff()
        ydiff = differentiate(self.yt,x).evaluateDiff()
        zdiff = differentiate(self.zt,x).evaluateDiff()
        return list([xdiff,ydiff,zdiff])

    def vecInt(self,lima,limb):
        # x = param
        xint = integrate(lima,limb,self.xt).evaluateInt()
        yint = integrate(lima,limb,self.yt).evaluateInt()
        zint = integrate(lima,limb,self.zt).evaluateInt()
        return list([xint,yint,zint])

    def Vlis(self):
        return list([self.xt,self.yt,self.zt])

    def scalarMultiplication(self,scalar):
        xstr = '{}*'.format(scalar) + self.xt
        ystr = '{}*'.format(scalar) + self.yt
        zstr = '{}*'.format(scalar) + self.zt
        return vectorFunction(xstr,ystr,zstr)

    def vectorDotProduct(self,vectorObject,value=None):
        vlis = vectorObject.Vlis()
        xstr = "{}*{}".format(vlis[0],self.xt)
        ystr = "{}*{}".format(vlis[1],self.yt)
        zstr = "{}*{}".format(vlis[2],self.zt)
        dotStr = xstr + ' + ' + ystr + ' + '+ zstr
        if(value != None):
            x = value
            return ndecimalPlaces(eval(dotStr),3)
        return dotStr


class Quaternion:
    """

     Quaternion is a system of numbers of the form q = a + x*i + y*j + z*k such that the following is defined
     i**2 = j**2 = k**2 = i*j*k = -1 where a,x,y,z are members of the Real numbers. Quaternion class takes a list
     or iterable of 4 real numbers. For Quartenions commutativity does not hold. Meaning given two quaternions p
     and q, p*q != q*p. Quaternion Object is declared as follows: 
     
     Example:                                                  
     import Quaternion as Q                                   
     q = Q([1,2,3,4])                                         
     q.modulus() # returns 5.4772                             
     p = Q([5,6,7,8])                                         
     r = p*q #returns -60.0 + 12.0i + 30.0j + 24.0k                                            
     rinv = q*p # returns -60.0 + 20.0i + 14.0j + 32.0k

     The 3d rotation vector can be used as with LOperator:                                        
        vector = [1,2,4] #     a list of numbers initialized. 
        Rotated = q.LOperator(vector) 
        #returns [28. 32. 82.].
      
     Other methods under the Quaternion class:
     1. conjuagte()
     2. unitQuaternion
     3. inverse etc.
     Doc will be made available in later releases
        """

    def __init__(self,qArr):

        self.q = np.array(qArr)
        self.len = len(qArr)

    def __repr__(self):
        expression = "{}{}i{}j{}k".format(ndecimalPlaces(self.q[0],3),negativeDetector(ndecimalPlaces(self.q[1],3)),negativeDetector(ndecimalPlaces(self.q[2],3)),negativeDetector(ndecimalPlaces(self.q[3],3)))
        return expression

    def __add__(self,quaternionObject):
        for j in range(len(self.q)):
            self.q[j] += quaternionObject.q[j]
        return Quaternion(self.q)

    def __sub__(self,quaternionObject):
        for k in range(len(self.q)):
            self.q[k] -= quaternionObject.q[k]
        return Quaternion(self.q)

    def __mul__(self,quaternionObject):
        q0 = self.q[0]*quaternionObject.q[0] - np.dot(self.q[1:4],quaternionObject.q[1:4])
        cross3 = np.cross(self.q[1:4],quaternionObject.q[1:4])
        q123 = self.q[0]*quaternionObject.q[1:4] + quaternionObject.q[0]*self.q[1:4] + cross3
        return Quaternion([q0,q123[0],q123[1],q123[2]])

    def __truediv__(self,quaternionObject):
        q0 = self.q[0]*quaternionObject.conjugate().q[0] - np.dot(self.q[1:4],quaternionObject.conjugate().q[1:4])
        cross3 = np.cross(self.q[1:4],quaternionObject.conjugate().q[1:4])
        q123 = self.q[0]*quaternionObject.conjugate().q[1:4] + quaternionObject.conjugate().q[0]*self.q[1:4] + cross3
        return Quaternion(1/(self.modulus()**2) * np.array([q0,q123[0],q123[1],q123[2]]))

    def __eq__(self,qObj):
        return (self.q[0] == qObj.q[0]) and (self.q[1] == qObj.q[1]) and (self.q[2] == qObj.q[2]) and (self.q[3] == qObj.q[3])
    
    def __gt__(self,qObj):
        return (self.q[0] > qObj.q[0]) and (self.q[1] > qObj.q[1]) and (self.q[2] > qObj.q[2]) and (self.q[3] > qObj.q[3])

    def __lt__(self,qObj):
        return (self.q[0] < qObj.q[0]) and (self.q[1] < qObj.q[1]) and (self.q[2] < qObj.q[2]) and (self.q[3] < qObj.q[3])

    def __le__(self,qObj):
        return (self.q[0] <= qObj.q[0]) and (self.q[1] <= qObj.q[1]) and (self.q[2] <= qObj.q[2]) and (self.q[3] <= qObj.q[3])

    def __ge__(self,qObj):
        return (self.q[0] >= qObj.q[0]) and (self.q[1] >= qObj.q[1]) and (self.q[2] >= qObj.q[2]) and (self.q[3] >= qObj.q[3])

    def modulus(self):
        return norm(self.q,self.len)

    def unitQuartenion(self):
        res = 1/(self.modulus()) * self.q
        return Quaternion(res)

    def conjugate(self):
        self.q[1:4] = -1*self.q[1:4]
        return Quaternion([q for q in self.q])

    def inverse(self):
        self.q[1:4] = -1*self.q[1:4]
        lisQ = list(1/self.modulus()**2 * self.q)
        return Quaternion(lisQ)
    
    def LOperator(self,qVector):
        qVector = np.array(qVector)
        q0 = self.q[0]
        qlis3 = self.q[1:4]
        Lqv = (q0**2 - norm(qlis3,len(qlis3))**2) * qVector + (2*np.dot(self.q[1:4],qVector)*qVector + 2*q0*(np.cross(self.q[1:4],qVector)))
        return Lqv

class symmetricMatrix:

    def __init__(self,matrix):

        self.m = np.array(matrix)
        self.len = len(matrix)

    def __repr__(self): 
        i = 0
        expression_lis = []
        while(i < len(self.m)):
            expression_lis.append("{}\t{}\n".format(self.m[i][0],self.m[i][1]))
            i += 1
        expression = expression_lis[0] + expression_lis[1]
        return expression

    def isSymmetric(self):
        comparison = self.m == np.transpose(self.m)
        return comparison.all()

    def SpectralDecomposition(self):
        eigVals,eigVectors = LA.eig(self.m) 
        expression = "A = {}{}\u2081{}\u2081T {}{}\u2081{}\u2081T \n".format(negativeDetector(eigVals[0]),'v','v',negativeDetector(eigVals[1]),'v','v')
        expression += "v\u2081 = {}\nv\u2082 = {}\n".format(eigVectors[0],eigVectors[1])
        print(expression)
        return (eigVectors,eigVals)

    def OrthogonalVariable(self):
        a = self.SpectralDecomposition()[0]
        a1 =  np.array(a[0])
        a2 =  np.array(a[1])
        return np.transpose(np.array([a1,a2]))

    def diagonalize(self):
        x = self.SpectralDecomposition()[1]
        diagonalMatrix = np.array([[x[0],0],[0,x[1]]])
        return symmetricMatrix(diagonalMatrix)


    def Quadraticform(self,vector):

        xT = np.array(vector)
        Ax = np.array([np.dot(self.m[i],vector) for i in range(len(vector))])
        Qx = np.dot(xT,Ax)
        return Qx


def dataGenerator(datalen,numRange):
    r = random.randint
    return [r(-(i+1)*numRange/(i+1),(i+1)*numRange/(i+1)) for i in range(datalen)]


class Autoregression:

    def __init__(self,xdata,ydata):

        self.xd = np.array(xdata)
        self.yd = np.array(ydata)
        self.b0 = 0
        self.b1 = 0
        if(len(xdata) == len(ydata)):
             self.mem = len(xdata)
             self.dataState = True
        else:
            self.dataState = False

    def autoUpdate(self,xnum,ynum):
        if(self.dataState):
            xdata = list(self.xd)
            xdata.pop(0)
            xdata.append(xnum)
            self.xd = np.array(xdata)
            ydata = list(self.yd)
            ydata.pop(0)
            ydata.append(ynum)
            self.yd = np.array(ydata)
            return None
        else:
            print(" Length of x and y are not the same")
            return None

    def __repr__(self):
        expression = "MemoryLength = {} \nxData = {} \nyData = {}".format(self.mem,self.xd,self.yd)
        return expression
    
    def LinearRegression(self):
        xmean = np.dot(self.xd,np.ones(self.mem))/self.mem
        ymean = np.dot(self.yd,np.ones(self.mem))/self.mem
        self.b1 = np.sum((self.xd - xmean*np.ones(self.mem)) * (self.yd - ymean*np.ones(self.mem))) / np.sum((self.xd - xmean*np.ones(self.mem))**2)
        self.b0 = ymean - self.b1*xmean
        print("y = {}{}*x".format(negativeDetector(ndecimalPlaces(self.b0,3)),negativeDetector(ndecimalPlaces(self.b1,3))))
        return [self.b0,self.b1]

    def predict(self,testValue):
        return ndecimalPlaces(self.b0 + self.b1*testValue,3)

    def RegressionPlot(self,Xlabel="x-axis",Ylabel="y-axis"):
        plt.scatter(self.xd,self.yd,s=10)
        x = np.linspace(-10,10,40)
        y = self.b0 + self.b1*self.xd
        plt.xlabel(Xlabel)
        plt.ylabel(Ylabel)
        plt.plot(self.xd,y)
        plt.show()
        return None

    def RmsError(self):
        y = self.b0 + self.b1*self.xd
        return math.sqrt(np.sum((self.yd - y)**2)/self.mem)


def iterative_num_function(initnum,iterlen,iterString):
    flis = [initnum]
    for i in range(iterlen):
        x = i+1
        flis.append(eval(iterString))
    return flis

def iterative_function(initnum,iterlen,iterString):
    '''Use the the list with name "a" indexed with the letter "j" to make program work'''
    # dictObj = globals[listName] 
    # dictObj = [initnum]
    a = [initnum]
    for j in range(iterlen):
        x = j+1
        a.append(eval(iterString))
    return a


def nth_fourier_coefficients(nNumber,funcExpression):
    pi = math.pi
    cosExp = '{}*cos({}*x)'.format(funcExpression,nNumber)
    sinExp = '{}*sin({}*x)'.format(funcExpression,nNumber)
    an = 1/pi * integrate(-pi,pi,cosExp).evaluateInt()
    bn = 1/pi * integrate(-pi,pi,sinExp).evaluateInt()
    print('Fourier coefficients for n = {}\na{} = {} \nb{} = {} '.format(nNumber,nNumber,an,nNumber,bn))
    return [an,bn]

def Fourier_Transform(svalue,expression):
    sRe = svalue.Re
    sIm = svalue.Im
    ImStr = '-1* {} * sin({}*x)*exp(-{}*x)'.format(expression,sIm,sRe)
    ReStr = '{} * cos({}*x)*exp(-{}*x)'.format(expression,sIm,sRe)
    print(ImStr)
    print(ReStr)
    return complex_function(ReStr,ImStr).complex_int(-100000000000000000000000000000000000000000000,100000000000000000000000000000000000000000000)


def periodic_function(xlis,tlis,frequency=1,wavelength=1,amplitude=1):
    ''' Please ensure all values are in standard units to get correct answers'''
    x = np.array(xlis)
    t = np.array(tlis)
    f = frequency
    w = wavelength
    A = amplitude
    y = A * np.sin(2*math.pi*(f*t - x/w))
    plt.plot(x,y,color='g')
    plt.plot(t,y,color='b')
    plt.show()

def stationary_interference(amp,freq,wavelen,phase1,phase2):
    phasediff = phase1 - phase2
    xlis = np.linspace(-amp,amp,num=1000)
    tlis = np.linspace(0,100,num=1000)
    y = 2*amp*math.cos(phasediff/2) * np.cos(2*math.pi*(xlis/wavelen - freq*tlis) + phasediff/2)
    y1 = amp*np.cos(2*math.pi*(xlis/wavelen - freq*tlis))
    y2 = amp*np.cos(2*math.pi*(xlis/wavelen - freq*tlis) + phasediff)
    plt.plot(xlis,y,color='b')
    plt.plot(xlis,y1,color='g')
    plt.plot(xlis,y2,color='r')
    plt.title('y = y1 + y2 y=blue y1=green y2=red')
    plt.show()


def scientificNotation(num):
    signed = False
    returnExpression = ''
    powlen = 0
    if(num < 0):
        signed = True
        num = -num
        
    if(num > 0 and num < 1):
        # powlen = 0
        while(True):
            num *= 10
            powlen += 1
            if(num >= 1):
                break
        
        returnExpression = '{} * 10**{}'.format(num,-powlen)


    elif(num > 1):

        try:

           while(True):
              num /= 10
              powlen += 1
              if(num >= 1 and num < 10):
                  break
           returnExpression = '{} * 10**{}'.format(num,powlen)

        except(OverflowError):
            
            numstr = str(num)
            nlen = len(numstr)
            numlis = numstr.split('.')
            if(len(numlis) == 1):
                 numlis.append('0')
            gnum = int(numlis[0])
            gnumlen = len(str(gnum))
            a = gnum/10**(gnumlen-1)
            returnExpression = '{}{} * 10**{}'.format(a,numlis[1],(gnumlen-1))

    
    if(signed):
        return '-' + returnExpression
    else:
        return returnExpression

def DrawCircle(center,radius):
    pi = math.pi
    angleList = np.linspace(-pi,pi,num=2000)
    x = center[0] * np.ones(len(angleList)) + np.ones(len(angleList)) * radius * np.cos(angleList) 
    y = center[1] * np.ones(len(angleList)) + np.ones(len(angleList)) * radius * np.sin(angleList)
    plt.scatter(x,y,s=1)
    plt.show()

def DrawEllipse(center,semiMajorAxis,semiMinorAxis):
    pi = math.pi
    angleList = np.linspace(-pi,pi,num=2000)
    y = center[1] * np.ones(len(angleList)) + np.ones(len(angleList)) * semiMinorAxis * np.sin(angleList)
    x = center[0] * np.ones(len(angleList)) + np.ones(len(angleList)) * semiMajorAxis * np.cos(angleList)
    plt.scatter(x,y,s=1)
    plt.show()

def eccentricity(semMajAxis,semMinAxis):
        return math.sqrt(1 - (semMinAxis/semMajAxis)**2)
    # elif(figuretype == 'hyperbola'):
    #     return math.sqrt(())

def DrawHyperbola(center,smajAxis,sminAxis):
    pi = math.pi
    angleList = np.linspace(-pi,pi,num=2000)
    x = center[0] * np.ones(len(angleList)) + np.ones(len(angleList)) * smajAxis * np.cosh(angleList)
    y = center[1] * np.ones(len(angleList)) + np.ones(len(angleList)) * sminAxis * np.sinh(angleList)
    plt.scatter(x,y,s=1)
    plt.show()

def DrawParabola(parabolaType,focus,avalue,includeDirectrix=True):
    x = []
    if(parabolaType == 'right'):
        x = list(np.linspace(focus[0],1000,num=10000))
    elif(parabolaType == 'left'):
        x = list(np.linspace(focus[0],-1000,num=10000))
    
    for j in range(len(x)):
        x.append(x[j])
    
    
    ypos = list(focus[1] + 2 * np.sqrt(avalue * (x - focus[0]*np.ones(len(x)))))
    yneg = list(focus[1] - 2 * np.sqrt(avalue * (x - focus[0]*np.ones(len(x)))) )
    

    x = np.array(x)
    ypos = np.array(ypos)
    yneg = np.array(yneg)

    plt.scatter(x,ypos,s=1,color='r')
    plt.scatter(x,yneg,s=1,color='r')
    plt.scatter([avalue + focus[0]],[focus[1]],s=5,color='b')
    fig,ax = plt.subplots()
    if(includeDirectrix):
        fx = focus[0] - avalue
        line = plt.Line2D([-fx,-fx],[-100,100],color='b',linewidth=2,alpha=0.8,clip_on=False)
        ax.add_artist(line)
    # plt.scatter([avalue + focus[0]],[focus[1]],s=5,color='b')
    plt.show()

def prime_detector(factlis):
    reslis = []
    for i in range(len(factlis)):
        ftes = prime_factorization(factlis[i])
        if(len(ftes) == 0):
            reslis.append(factlis[i])

    return reslis



def prime_factorization(num):
    factlis = []
    j = num-1
    while(j > 1):
        if(num%j == 0):
            factlis.append(j)
        j -= 1
    
    return factlis


def prime_powers(num,prime):
    power = 0
    while(True):
        if(num%prime == 0):
            power += 1
            num = num/prime

        else:
            break

    return power


def totient(power_list):
    totnp = np.array([power_list[k][0]**(power_list[k][1]-1) * (power_list[k][0]-1) for k in range(len(power_list)) ])
    return np.product(totnp)


def prime_multiple(n):
    p = prime_factorization(n)
    # print(p)
    pdetector = prime_detector(p)
    powerlist = []
    for j in range(len(pdetector)):
        powerlist.append([pdetector[j],prime_powers(n,pdetector[j])])

    expre = ''
    for i in range(len(powerlist)):
        if(i == len(powerlist)-1):
            expre += '{}^{}'.format(powerlist[i][0],powerlist[i][1])
        else:
            expre += '{}^{} * '.format(powerlist[i][0],powerlist[i][1])

    tot = totient(powerlist)

    return [expre,tot]

def additive_inverse(num,mod):
    ''' Suppose a is a positive integer then for some positive integer n such that n >= a the additive inverse for a
    in mod n is b = n - a '''
    if(num > mod):
        print("Additive inverse of {} in mod {} is not defined since it\'s negative\n".format(num,mod))
        return None
    else:
        return mod - num

def multiplicative_inverse(num,mod):
    """ if a is a positive integer and for any given positive integer n such that gcd(a,n) = 1, then 
    there exist an integer b such that a * b = 1 mod(n) """

    iteration = 0
    state = False
    gcd = math.gcd(num,mod)
    if(gcd == 1):
        state = True
    if(state):
        while(state):
            iteration += 1
            if(((iteration*mod)+1)%mod == 1 and ((iteration*mod)+1)%num == 0 ):
                break
        print('iterations = ',iteration)
        return int(((iteration*mod)+1)/num)
    else:
        print("The number {} has no multiplicative in modulo {} since gcd({},{}) = {}".format(num,mod,num,mod,gcd),end="\n")
        return None

def euler_approximation(fxy,num_steps,h=0.1,IV=(0,0)):
    """ Euler's approximation is used to solve ODE's of the form y' = f(x,y)
    for some function f(x,y)"""
    x = IV[0]
    y = IV[1]
    yf = 0
    for i in range(num_steps):
        yf = y + h*eval(fxy)
        x += h
        y = yf

    return [x,y]

def randomTester(choices,numTrials):
    results = []
    choiceNums = []
    frequency = 0
    
    for k in range(numTrials):
        index = r(0,len(choices)-1)
        results.append(choices[index])

    for i in range(len(choices)):
        for j in range(numTrials):
            if(choices[i] == results[j]):
                frequency += 1
        choiceNums.append(frequency)
        frequency = 0
    
    probabilities = [choiceNums[i]/numTrials for i in range(len(choices))]
    del results

    return [choices,probabilities]


def randNumber(choices,numTrials):
    randNumberlis = []
    for i in range(1000):
        rt = randomTester(choices,numTrials)
        trueProb = 1/len(choices)
        probnp = np.array(rt[1]) 
        probabilityError = (1 - np.mean(np.abs(trueProb * np.ones(len(choices)) - probnp)))
        randNumberlis.append(probabilityError)

    return rms(randNumberlis)


def DiceGame(initScore):
    myScore = initScore
    computerScore = initScore
        
    while(True):

        myChoice = r(1,6) #int(input("Roll your dice: "))
        computerChoice = r(1,6)

        print("My dice number {}\n".format(myChoice))

        print("Computer's dice number: {}\n".format(computerChoice))
        if(myChoice > 6 or myChoice < 0):
            print("The number {} is not on a die.\nPlease try again\n".format(myChoice))
            continue
        if(myScore == 0 or computerScore == 0):
            break
            
        if(computerChoice > myChoice):
            myScore -= 1
        elif(computerChoice < myChoice):
            computerScore -= 1
        else:
            myScore = myScore
            computerScore = computerScore
        time.sleep(2)
        
    return [myScore,computerScore]

import time
def randomNumber():
    m = 2**50 - 1
    t = time.time()
    x0 = 381473871394
    a = int(t)
    c = int((t - a)* 10**6)
    print(a,' ',c,' ',t,' ',m)
    return ((a*x0 + c)%m)/m


# def VonNeumannGenerator(num_digits):
#     t = time.time()
#     strt = str(int(t))
#     lent = len(strt)

def GaussianElimination(ArrayOfNVectorsWithSolution):
    ''' The ArrayOfNvectorsWithSolution should include the solution number for each equation
    eg. A system of linear equations given by:

    x + 2y + 4z = 10
    2x + 7y + 10z = 24
    x - y + 2z = -10

    In matrix form will be written as 
    1  2   4      x      10 
    2  7  10  *   y   =  24  can be written collectively as 
    1 -1 -10      z     -10  
    
    [ [1,2,4,10] , [2,7,10,24] , [1,-1,-10,-10] ] 

    before being passed as an argument to the Gaussian elimination function

    Solved using the equation: R[n] = R[k][k] * R[n] - R[n][k] * R[k]

    n iterates from k+1 to L where L is the length of one vector in the

    array of vectors R. k iterates from 0 to L - 1

    '''
    
    iternum = 0
    # j = 0
    x = []
    R = np.array(ArrayOfNVectorsWithSolution)
    for k in range(len(R)-1):
        for n in range(k+1,len(R)):
            R[n] = R[k][k] * R[n] - R[n][k] * R[k]
    # print(R)
    # a = np.array([R[j][0:len(R[0])-1] for j in range(len(R[0])-1)])
    # b = np.array([R[k][len(R[0])-1] for k in range(len(R[0])-1)])
    # x = list(np.zeros(len(R[0])))
    # n = len(R[0])-2
    # x[len(R[0])-1] = b[n]/a[n][n]
    
    # sumTotal = 0

    # for k in range(len(R[0])-2):
    #     for j in range(0,k-1):
    #         sumTotal += a[n-k][n-j] * x[n-j]
    #     x[n-k] = (b[n-k] - sumTotal)/a[n-k][n-k]
    #     sumTotal = 0


    return R


from graphics import * 

def randomWalk1d(position=0):
    ''' Random walk 1d simulates a random walk in 1 dimension. It returns the time interval between when the walk starts at 
    given position and returns back to the position. ''' 
    step = r(1,2)
    if(step == 2):
        step = -1
    return step+position

def iteratedRandomWalk1d():
    startTime = time.time()
    position = 0
    try:
        while True:
            position = randomWalk1d()
            if(position == 0):
                interval = time.time() - startTime
                break
        return interval

    except(KeyboardInterrupt):
        print('\n',position)
        return (time.time() - startTime)




def randomWalk2d(positionxy=(0,0),step_width=5):
    win = GraphWin('random walk 2d',1000,1000,autoflush=True)
    win.setBackground('white')
    startPoint1 = Point(positionxy[0],positionxy[1])
    c = Circle(startPoint1,5)
    c.setFill('red')
    c.draw(win)
    num_steps = 0
    step_size = step_width

    startTime = time.time()
    text = Text(Point(900,50),'Time: {}s\n steps: {}\nstep_size: {}'.format('0.000',num_steps,step_size))
    
    text.setFill('green')
    text.setSize(10)
    text.setStyle('italic')
    text.setFace('arial')
    text.draw(win)

    while True:
        startPoint = Point(positionxy[0],positionxy[1])
        startPointxy = (positionxy[0],positionxy[1])
        stepx = r(1,2)
        stepy = r(1,2)
        if(stepx == 1):
            stepx = step_size
        if(stepy == 1):
            stepy = step_size
        if(stepx == 2):
            stepx = -step_size
        if(stepy == 2):
            stepy = -step_size
        num_steps += 1
        position = Point(positionxy[0]+stepx,positionxy[1]+stepy)
        positionxy = (positionxy[0]+stepx,positionxy[1]+stepy)

        line = Line(startPoint,position)
        line.setFill('blue')
        win.plot(startPointxy[0],startPointxy[1],'red')
        win.plot(positionxy[0],positionxy[1],'yellow')

        startPoint = position
        line.draw(win)

        interval = time.time() - startTime
        #time.sleep(0.05)
        text.setText('start position: {}\nTime: {}s\n steps: {}\nstep_size: {}\nCurrent position: {}'.format((startPoint1.getX(),startPoint1.getY()),interval,num_steps,step_size,positionxy))
        if(position == startPoint1):
            win.close()
        win.update()
       
    return None

    

def projectile(u=0,theta=math.pi/6):

    cos = math.cos
    sin = math.sin
    tan = math.tan

    g = 9.81
    startTime = time.time()
    epochTime = time.time()
    win = GraphWin('Projectile motion',1000,1000,autoflush=True)
    centre = np.array((0,0))
    circ = Circle(Point(centre[0],centre[1]),radius=10)
    circ.setFill('blue')
    circ.draw(win)

    hmax = (u * sin(theta))**2/(2*g)
    timeMaxHieght = (u*sin(theta))/(2*g)
    Range = (u**2 * cos(2*theta))/g
    print(Range)
    heightState = False

    while True:
        Duration = time.time() - epochTime

        dt = time.time() - startTime
        uy = u*sin(theta)
        dx = u*cos(theta)*dt 
        dy = uy*dt + 0.5 * g * dt**2
        startTime = time.time()

        uy = uy - g * dt

        if(Duration >= timeMaxHieght):
            heightState = True

        while(heightState):

            dt = time.time() - startTime
            uy = u*sin(theta)
            dx = u*cos(theta)*dt 
            dy = uy*dt - 0.5 * g * dt**2
            startTime = time.time()

            uy = uy + g * dt
            circ.move(dx,dy)
            # circ.draw(win)
            win.update()

        circ.move(dx,dy)
        win.update()



def nPredict(n,tvector,yvector):
     """ 
    
     The nPredict function capitalizes on the equation y = x0 + x1*t + x2*t**2 + ... + xn*t**n
     as it sole predictor with y being the variable to predict and t being the  independent variable
     taken as an input. The values x0,x1,x2,...,xn are the appopriate coefficients to be chosen to fit
     the y  and the t.
    
     """
     tlis1 = []
     tlis2 = []
     x = np.random.randint(1,100,len(tvector)+1)
     print(x)
     t1 = tvector[0]
     t2 = tvector[1]

     for i in  range(len(tvector)+1):
         tlis1.append(t1**(i))
         tlis2.append(t2**(i))
     
     tlis1 = np.array(tlis1)
     tlis2 = np.array(tlis2)
     dt = tlis1 - tlis2
     dy = yvector[0] - yvector[1] 
     dx = (dy/norm(dt,len(dt))**2 * dt,) - x

     x = x + dx

     return x

    
def gravity(accleration,initialHeight):

    win = GraphWin('Gravity simulation',1000,1000)
    w = 250
    l = 250
    p1 = Point(50,600)
    p2 = Point(550,600)
    base = Rectangle(p1,p2)
    base.setFill('black')
    base.draw(win)

    centre = (500,10)

    while(True):
        circ = Circle(centre,radius=20)
        circ.setFill('blue')
        circ.draw (win)

        

def rms(iterable,dtype=float):
    # if(dtype != float or dtype!=int):
    #     print("rms is not defined for an iterable of type {} ".format(type(iterable[0])))
    #     return None
    iterable = np.array(iterable)
    sumSquares = np.sum(iterable**2)
    return math.sqrt(sumSquares/len(iterable))

def wholeNumberDetector(num1,num2):
    wholeNum = int(str(num1/num2).split('.')[0])
    remainder = (num1/num2) - wholeNum
    if(remainder > 0):
        return wholeNum
    else:
        return [num1,num2]



class fraction:

    def __init__(self,numerator,denominator):

        gcd = math.gcd(numerator,denominator)
        self.n = int(numerator/gcd)
        self.d = int(denominator/gcd)

    def __repr__(self):
        return'{}/{}'.format(self.n,self.d)

    def __add__(self,fractionObject):
        if(fractionObject.d == self.d):
            num = self.n + fractionObject.n
            den = self.d
            gcd = math.gcd(num,den)
            return fraction(int(num/gcd),int(den/gcd))
        else:
            den = self.d * fractionObject.d
            num = (self.d * fractionObject.n) + (self.n * fractionObject.d)
            gcd = math.gcd(num,den)
            return fraction(int(num/gcd),int(den/gcd))

    def __mul__(self,fractionObject):
        den = self.d * fractionObject.d
        num = self.n * fractionObject.n
        gcd = math.gcd(den,num)
        return fraction(int(num/gcd),int(den/gcd))

    def __truediv__(self,fractionObject):
        num = fractionObject.d * self.n
        den = fractionObject.n * self.d
        gcd = math.gcd(num,den)
        return fraction(int(num/gcd),int(den/gcd))

    def __pow__(self,power):

        if(str(type(power)) != "<class 'int'>"):
            print("Type {} is not supported ".format(type(power)))
            return None

        else:
            return fraction(self.n**power,self.d**power)


def argmax(vector):
    
    v = np.array(vector)
    newVector = np.zeros(len(v))
    maxValue = np.max(v)
    where = np.where(v == np.max(v))
    whereLen = len(where[0])

    for k in range(len(v)):

        if(v[k] == maxValue):
            newVector[k] = 1/whereLen
            
        elif(v[k] != maxValue):
            newVector[k] = 0
       

    return newVector


def softmax(array,base=np.exp(1)):

    array = np.array(array)
    arrlen = len(array)
    basePowerSum = np.sum(base**(array))
    returnArr = base**array/basePowerSum
    return returnArr


def Relu(vector):
    v = np.array(vector)
    return np.maximum(v,0)

def LeakyRelu(vector,alpha=0.0):
    v = np.array(vector)
    return np.maximum(v,alpha*v)

def tanh(vector):
    v = np.array(vector)
    return np.tanh(v)

def sigmoid(vector):
    v = np.array(vector)
    return 1/(1 + np.exp(-v))




