#!/usr/bin/env python3
import numpy as np
from functools import reduce

key=30
# Creates an empty 2-Dimensional array (100 by 10)
array1 = np.empty([5,10]).tolist()
for i in range(0,5):
    array1[i] = np.arange(0,10).tolist()
print(type(array1))
for i in range(0,5):
    var = reduce(lambda a,b: a.union(b), array1[max(0,i-key):i+1], set())
    print(var)
    #print(array1[max(0,i-key):i+1])
