import math
import random
import numpy

# The classical one-qbit states.
ket0 = numpy.array([1+0j,0+0j])
ket1 = numpy.array([0+0j,1+0j])

# Write this function. Its input is a one-qbit state. It returns either ket0 or ket1
def measurement(state) :
    states = [ket0, ket1]
    weights = [abs(state[0])**2,abs(state[1])**2]
    return random.choices(states,weights)

# For large m, this function should print a number close to 0.64. (Why?)
# Because Law of Large Number take over, f() returns 1 with the probability of 0.64 and 0 with the
# probability of 0.36, so on average 0.64 is the result returned by the measurementTest345() function

def measurementTest345(m) :
    psi = 0.6 * ket0 + 0.8 * ket1
    print(measurement(psi))
    def f() :
        if (measurement(psi) == ket0).all() :
            return 0
        else :
            return 1
    acc = 0
    for i in range(m):
        acc += f()
    return acc/m

print(measurementTest345(100000))