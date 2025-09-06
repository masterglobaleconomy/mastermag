from scipy.sparse import csr_matrix
import numpy as np
from master_equation.Cram import CRAM
import matplotlib.pyplot as plt

A = np.matrix([[-1, 0 , 0, 0], [1, -1.2, 0, 0], [0, 1.2, -0.8, 0], [0, 0, 0.8, 0]])  # Macierz wejściowa
B = np.array([0.0, 0.0, 0.0, 0.0]) # Macierz diagonalna która wprowadzi element wzrostu  dA/dt = s*A

A += np.matrix(np.diag(B))

n0 = [1, 0, 0, 0] # initial values
dt = 10
print(A)
X = np.linspace(0, 10, 500)
Y1 = []
Y2 = []
Y3 = []
Y4 = []

cram = CRAM()

for dt in X:
    y = cram(A, n0, dt)
    print(y)
    Y1.append(y[0])
    Y2.append(y[1])
    Y3.append(y[2])
    Y4.append(y[3])

# print("X:")
# print(X)
# print("Y1:")
# print(Y1)

a = plt.scatter(X, Y1, s=0.5)
b = plt.scatter(X, Y2, s=0.5)
c = plt.scatter(X, Y3, s=0.5)
d = plt.scatter(X, Y4, s=0.5)
plt.legend((a, b , c, d), ('a', 'b', 'c', 'd'))
plt.show()
