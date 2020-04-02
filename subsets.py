import numpy as np

Set = [1,2,3,4]
set_len = len(Set)
i = 0
j = 0
k = 0

sub_order = []

def factorial(n):
    if n == 1:
       return n
    elif n == 0:
       return 1
    else:
       return n*factorial(n-1)

def combination(n,r):

    c = factorial(n)/(factorial(n-r)*factorial(r))
    return int(c)

while(i < set_len):
      sub_order.append(combination(set_len,i))
      i += 1

sub_order.append(1)
subsets = [{}]
singletons = []

while(k < set_len):
   subsets.append(set([Set[k]]))
   singletons.append(set([Set[k]]))
   k += 1
i = 0
while(i < len(singletons)):
     while(j < len(singletons)-i):
          subsets.append(singletons[j].union(singletons[i]))
          j += 1
     i += 1

print(subsets)


