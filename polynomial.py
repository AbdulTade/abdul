import numpy as np 
class polynomial:

    def __init__(self,coefficient_vector):

        self.cvector = np.array(coefficient_vector)

    def  __repr__(self):

        length = len(self.cvector)
        j = 0
        expression = ''
        while(j < length):

            expression += " {}*x^{} + ".format(self.cvector[j],j)
            j += 1
        return expression

    def poly_add(self,second_coeffient_vector):
        i = 0
        lis = []
        length = len(self.cvector)
        while(i < length):
             lis[i] = self.cvector[i] + second_coeffient_vector[i]
             i += 1
        return polynomial(lis)

    def poly_subtract(self,second_coeffient_vector):
        k = 0
        sub_lis = []
        length = len(self.cvector)
        while(k < length):
            sub_lis[k] = self.cvector[k] - second_coeffient_vector[k]
            k += 1
        return polynomial(sub_lis)

    def poly_multiplication(self,second_coeffient_vector):
        n = 0
        m = 0
        mult_lis = []
        term = []
        len1 = len(self.cvector)
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

        

        