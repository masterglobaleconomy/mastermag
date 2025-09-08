from gekko import GEKKO
m = GEKKO()
# variable array dimension
n = 3 # rows
p = 2 # columns
# create array
x = m.Array(m.Var,(n,p))
for i in range(n):
  for j in range(p):
            x[i,j].value = 2.0
            x[i,j].lower = -10.0
            x[i,j].upper = 10.0
# create parameter
y = m.Param(value = 1.0)
# sum columns
z = [None]*p
for j in range(p):
   z[j] = m.Intermediate(sum([x[i,j] for i in range(n)]))
# objective
m.Obj(sum([ (z[j]-1)**2 + y for j in range(p)]))
# minimize objective
m.solve()
print('x:', x)
print('z:', z)