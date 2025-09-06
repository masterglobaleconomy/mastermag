from pyomo.environ import *

"""
Limited to choose Knapsack Problem (limits in restrictions)
"""

A = ['hammer', 'wrench', 'screwdriver', 'towel']
b = {'hammer': 8, 'wrench': 3, 'screwdriver': 6, 'towel': 11}
w = {'hammer': 5, 'wrench': 7, 'screwdriver': 4, 'towel': 3}
r = {'hammer': 2, 'wrench': 2, 'screwdriver': 2, 'towel': 2}
W_max = 14
model = ConcreteModel()

model.x = Var( A, within=NonNegativeIntegers)


@model.Objective(sense = maximize)
def objective_rule(model):
    return sum( b[i]*model.x[i] for i in A)


@model.Constraint()
def weight_rule(model):
    return sum( w[i]*model.x[i] for i in A) <= W_max


@model.Constraint(A)
def number_rule(model, i):
    return model.x[i] <= r[i]


opt = SolverFactory('glpk')
result_obj = opt.solve(model, tee=True)
model.pprint()

# PRINT
print()
for i in A:
    print(model.x[i].value)


