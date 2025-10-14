from gekko import GEKKO    
import numpy as np
from master_equation.Cram import CRAM

# https://gekko.readthedocs.io/en/latest/model_methods.html

n = 4
p = 4
m = GEKKO()
A = m.Array(
    m.Var,
    (4, 4),
    value=0.0,
)

for i in range(n):
  for j in range(p):
            A[i,j].value = 0.1
            A[i,j].lower = -5.0
            A[i,j].upper = 5.0

# Teraz tzeba
type(A[0][0])
<class 'gekko.gk_variable.GKVariable'>
zrzutowac na float!!
type(A[0][0].value.value) is float
[x.value.value for x in A.flatten()]
np.array([x.value.value for x in A.flatten()], dtype=np.float32).reshape(4, 4)
# A = np.array([[-1, 0 , 0, 0], [1, -1.2, 0, 0], [0, 1.2, -0.8, 0], [0, 0, 0.8, 0]]) 
dt = np.float64(10)
cram = CRAM()
y0 =  np.array([1, 0, 0, 0], dtype=np.float64) 
y1 = cram(A, y0, dt)

print(y1)
print(type(y1))
print(y1 - y0)
Loss = np.mean((y1 - y0) ** 2)

# # change initial values
# x2.value = 5; x3.value = 5
# m.Equation(x1*x2*x3*x4>=25)
# m.Equation(x1**2+x2**2+x3**2+x4**2==40)
# m.Minimize(x1*x4*(x1+x2+x3)+x3)
# m.solve()
# print('x: ', x)
# print('Objective: ',m.options.OBJFCNVAL)