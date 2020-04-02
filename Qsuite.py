import math
# expression = '2*x**2+x-3*e(-x)'
# x = 3
# e = math.exp
# arr = list(expression)

# res = expression.split('+')

# print(res)

# print(eval(expression))

exp  =  math.exp
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

    def __repr__(self):

        return '({},{})'.format(self.Re,self.Im)

    def comp_scale(self,scalar):
        self.Re = self.Re*scalar
        self.Im = self.Im*scalar


    def modulus(self):

        return sqrt(self.Re**2 + self.Im**2)

    def argz(self):

        return math.atan(self.Im/self.Re)

    def conjugate(self):

        return complex_number(self.Re,-self.Im)

    def polar(self):

        val = (self.modulus(),self.argz())
        return val

    def complex_power(self,n):

        (modulus,argz) = self.polar()
        modulus = modulus**n
        argz = n*argz
        return complex_number(modulus*math.cos(argz),modulus*math.sin(argz))

   
    def complex_function(self,complex_expression):
        res = eval(complex_expression)
        return res


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
            expression += " {}*x^{} + ".format(self.cvector[j],j)
            j += 1
        return expression

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
  return k*vec2

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
        return np.mode(vectorArray)

    def std(self):
        return np.std(self.vec)

    def zscores(self):
        return 1/(self.std()) * (self.vec - np.ones*self.mean())



