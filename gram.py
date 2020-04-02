import numpy as np 
import math

vectorArray = np.array([[1,0,2],[3,5,0],[2,6,1]])
vectorlen = len(vectorArray)

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

res = orthonormalize(vectorArray)
ans = np.dot(res[1],res[2])
print(norm(res[1],3))
print(ans)
print(res)


