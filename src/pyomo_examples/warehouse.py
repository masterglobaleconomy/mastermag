from pyomo.environ import *
import pandas as pd
from Cwiczenie02 import haver_dist

"""
Warehouse location determination problem
"""

data = pd.read_csv('oryg_input.csv')
print(data.head())
print(list(data.name))


model = ConcreteModel(name="Warehouse")

W = list(data.name)
C = list(data.name)

d = {}
lW = len(W)
for i in range(lW):
    for j in range(i, lW):
        #if i != j:
            #print(W[i], W[j])
            d[(W[i], W[j])] = haver_dist.haversine_dist(data[data.name == W[i]][['latitude', 'longitude']].values[0], data[data.name == W[j]][['latitude', 'longitude']].values[0])
print(d)
#
# d = {('Harlingen', 'NYC'): 1956, ('Harlingen', 'LA'): 1606, ('Harlingen', 'Chicago'): 1410, \
#      ('Harlingen', 'Houston'): 330, ('Memphis', 'NYC'): 1096, ('Memphis', 'LA'): 1792, \
#      ('Memphis', 'Chicago'): 531, ('Memphis', 'Houston'): 567, ('Ashland', 'NYC'): 485, \
#      ('Ashland', 'LA'): 2322, ('Ashland', 'Chicago'): 324, ('Ashland', 'Houston'): 1236 }
P = 8

model.x = Var(W, C, within=NonNegativeReals, bounds=(0,1), initialize = 0)
model.y = Var(W, within=Binary)

@model.Constraint(C)
def one_per_cust_rule(model, c):
    return sum(model.x[w,c] for w in W) == 1

@model.Constraint(W, C)
def warehouse_active_rule(m, w, c):
    return m.x[w,c] <= m.y[w]

@model.Constraint()
def num_warehouses_rule(m):
    return sum(m.y[w] for w in W) <= P

def obj_rule(m):
    return sum(d[w,c]*m.x[w,c] for w, c in d.keys())
model.obj = Objective(rule=obj_rule)

SolverFactory('glpk').solve(model)

model.y.pprint()
#model.x.pprint()
