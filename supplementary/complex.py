


# To prepare for our Python project work, go through this tutorial in detail, executing each line in the Python interpreter, making sure that you understand what it's doing.



### COMPLEX NUMBERS ###

# Python has a standard library, called cmath, for math with complex numbers. I think that we never use cmath. We mainly use NumPy, which handles not just complex numbers but also linear algebra with complex numbers. (If the following import doesn't work, then install the library on your machine and try again.)
import numpy

# Here are some basics with complex numbers.
chi = numpy.array(3 - 2j)
chi # chi
chi.real # real part of chi
chi.imag # imaginary part of chi
abs(chi) # norm of chi (the modulus)
numpy.conj(chi) # conjugate
numpy.exp(chi) # for chi = a+bi, return e^a(cosb)-ie^a(sinb)
psi = numpy.array(7 + 4j)
chi + psi # add component-wise
chi - psi # subtract component-wise
chi * psi # multiply
chi / psi # divide

# Here are some mathematically ill-formed questions. But NumPy has clear answers to them. Can you figure out its interpretation?
chi > psi
chi < psi # compare the norm / modulus

# What does the following block of code do? Draw a picture. Does the final answer make sense?
m = 10
chis = [numpy.exp(1j * 2 * numpy.pi * t / m) for t in range(0, m)] # e^it\pi / m loop over m=0-9
chis
sum(chis) # add them together

# Exercise A: Write a function rect() as follows. It takes as input a real number r and a real number t. It returns the complex number r e^(i t). Write short demo code to verify that your rect() behaves correctly. Print your rect() code, demo code, and demo results, so that the grader is convinced that your code works.

def rect(r,t) :
    x = r*numpy.cos(t)
    y = r*numpy.sin(t)
    chi = x+y*1j
    return chi

print(rect(3,2))

### VECTORS ###

# Vectors (column matrices) act much like Python lists.
ketPsi = numpy.array([1, 2, 3, 4 + 5j])
ketPsi # array of 5
ketPsi[0] # first item in array
ketPsi[-1] # last item in array

# Basic vector arithmetic.
(2 + 1j) * ketPsi # multiply 2+1j with every component
ketPhi = numpy.array([3 - 1j, 3, 2 + 2j, 0])
ketPsi + ketPhi # adding

# numpy.dot() transposes and multiplies but does not conjugate. To compute the Hermitian inner product, you must explicitly conjugate the first argument to numpy.dot().
numpy.dot(ketPsi, ketPsi)
numpy.dot(numpy.conj(ketPsi), ketPsi)


### GENERAL MATRICES ###

# To the user, a matrix acts as if it's a list of rows.
a = numpy.array([[1, 3 - 2j, 4], [7j, 5, 1j]])
a
a[0]
a[1]

# To get an individual entry, both options below work, but apparently the second option is more efficient.
a[0][1]
a[0, 1]

# To grab a column, you can use : as row index.
a[:, 0]

# Query the dimensions and the underlying data type.
a.shape # matrix size
a.dtype # matrix type
a[0].shape # size of first row
a[:, 0].shape # size of first column
a[0, 1].shape

# Basic algebra.
numpy.conj(a)
numpy.transpose(a)
a.T # also transpose
chi = 3 - 2j
chi * a
b = numpy.array([[2 - 4j, -3 + 1j, 4 - 2j], [3 + 7j, 5j, 2 - 1j]])
b
a + b
a - b
c = numpy.matmul(a, numpy.transpose(b)) # multiplication
c

# Multiplying a matrix by a vector.
ketPsi = numpy.array([1, 2, 3])
numpy.dot(a, ketPsi)



### MORE MATRICES ###

# Eigensystems and matrix invariants.
numpy.linalg.det(c)
numpy.trace(c)
(vals, vecs) = numpy.linalg.eig(c)
vals # eigenvalue
vecs # eigenvector
numpy.dot(c, vecs[:, 0])
vals[0] * vecs[:, 0] # similar to above
numpy.dot(c, vecs[:, 1])
vals[1] * vecs[:, 1] # similar to above

# Making a matrix of zeros. By default it's real, which we don't want.
c = numpy.zeros((3, 2))
c
c.dtype

# So we have to explicitly tell numpy to use the complex type that we're already using. Here's one way to do it.
c = numpy.zeros((3, 2), dtype=numpy.array(0 + 0j).dtype)
c
c.dtype

# Concatenating matrices. First atop-below, then left-right.
c = numpy.concatenate((a, b))
c
c.shape
c = numpy.concatenate((a, b), axis=1)
c
c.shape

# Exercise B: Write a function directSum() as follows. It takes two complex matrices as input and returns one complex matrix as output. Suppose the first input matrix A is p x q and the second input B is m x o.
# Then the output matrix is (p + m) x (q + o). It consists of A and B along the diagonal and zeros elsewhere. For example, if

#     A = [A00 A01 A02]
# is 1x3 and
#     B = [B00 B01]
#         [B10 B11]
# is 2x2, then the direct sum is the 3x5 matrix
#     [A00 A01 A02 0   0  ]
#     [0   0   0   B00 B01].
#     [0   0   0   B10 B11]
# As in Exercise A, print your directSum() code, demo code, and demo results.

def directSum(A,B) :
    C = numpy.zeros((A.shape[0]+B.shape[0],A.shape[1]+B.shape[1]), dtype=numpy.array(0+0j).dtype)
    for r in range(C.shape[0]) :
        for c in range(C.shape[1]) :
            try:
                C[r,c] = A[r,c]
            except IndexError:
                C[r,c] = 0
    
    for r in range(-1,-C.shape[0]+1,-1) :
        for c in range(-1,-C.shape[1]+1,-1) :
            try:
                C[r,c] = B[r,c]
            except IndexError:
                C[r,c] = 0
    print(C)

A = numpy.array([[1,2-2j,4,5+2j,3],[3,4-2.3j,4,6+7.1j,3]])
B = numpy.array([[1,2-.4j],[1+1j,1]])

print(rect(3,2))
directSum(A,B)