"""
Line perception test.
"""

import random
import neural
import numpy
from matplotlib import pyplot

def f(x): 
    return x

def is_above(x,y,f):
    if y>f(x): return 1
    else: return -1

test_set = []
for n in range(2000):
    x = random.uniform(-10,10)
    y = random.uniform(-10,10)
    test_set.append([x,y,is_above(x,y,f)])

perception = neural.Perceptron(2)
print 'before:', perception
perception.training(test_set)
print 'after:', perception

"""
#plot
x = numpy.linspace(-10,10)
y = map(f,x)
pyplot.plot(x,y)
color = [random.random() for n in range(len(test_set))]
for test in test_set:
    pyplot.scatter(test[0], test[1], c=color, alpha=.3)

pyplot.show()
"""