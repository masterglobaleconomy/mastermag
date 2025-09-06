

import pandas as pd
from pyomo.environ import *
from pyomo.dae import *
import timeit

start = timeit.default_timer()




class BUCKET:
    def __init__(self, capacity):
        self.capacity = capacity

    def __call__(self):
        return self.capacity

    def get(self):
        return self.capacity

class STEELGRADE:
    def __init__(self, steelgrade, maxCharges, combinable):
        self.steelgrade = steelgrade
        self.maxCharges = int(maxCharges)
        self.combinable = combinable

    def get_steelgrade(self):
        return self.steelgrade

    def get_maxCharges(self):
        return self.maxCharges

    def get_combinable(self):
        return self.combinable

    def print_fields(self):
        print(self.steelgrade, self.maxCharges, self.combinable)

class ORDER:
    def __init__(self, steelgrade, weight, forbidden, mandatory, cost, flows):
        self.steelgrade = steelgrade
        self.weight = float(weight)
        self.forbidden = forbidden
        self.mandatory = mandatory
        self.cost = cost
        self.flows = flows


    def get_stelgrade(self):
        return self.steelgrade

    def get_weight(self):
        return self.weight

    def get_forbidden(self):
        return self.forbidden

    def get_mandatory(self):
        return self.mandatory

    def get_cost(self):
        return self.cost

    def get_cost(self, i):
        return self.cost[i]

    def get_flows(self):
        return self.flows

    def print_fields(self):
        print(self.steelgrade, self.weight, self.mandatory, self.forbidden, self.cost, self.flows)



class READ:
    _counter = 0
    def __init__(self, name="defoult"):
        READ._counter += 1
        self.name = name
        self.id = READ._counter


        # PARAMETERS

        self.combinableCosts = [2000, 2349.836752, 2738.070813, 3153.550554, 3590.542004, 4045.47821, 4515.900532, 5000, 0.0]
        self.uncombinableCosts = [4500, 4523.809524, 4563.492063, 4619.047619, 4690.47619, 4777.777778, 4880.952381, 5000, 0.0]

        self.exoticPlanningBucketLengths = [3, 4, 7, 7, 7, 7]
        self.exoticPlanningAverageChargesPerDay = 48
        self.exoticPlanningBadlyCombiningAnalyses = ["KK60", "KM60", "K600", "LK50", "LK70", "LM10", "LT60", "LT70", "LT80",
                                                     "L460", "L500", "L600", "L620", "L630", "L660", "L760", "L770", "L810",
                                                     "L920", "L950", "L960", "L970", "TB70"]
        self.exoticPlanningWellCombiningAnalyses = ["KB30", "KK70", "KK80", "KM20", "KM30", "KM40", "KM50", "KM70", "KM80",
                                                   "KN20", "KN30", "KN40", "KN50", "KN70", "KR50", "KT10", "KT50", "KT70",
                                                   "K110", "K120", "K130", "K140", "K170", "K200", "K210", "K220", "K270",
                                                   "K320", "K340", "K350", "K360", "K370", "K380", "K390", "K450", "K710",
                                                   "K820", "LK90", "NB60", "N710", "TB10", "TB30", "TB40", "TB50", "T210",
                                                   "T220", "T230", "T250", "T690"]

        self.flowCostPoints = [-0.5, -0.2, 0, 0.3, 0.7, 1, 1.4, 2]
        self.flowCostSlopes = [-200, -8, -4, -1, 0, 1, 2, 4, 100]
        self.heatWeight = 298400 # * weight in kg

    def read_all(self, file_name_list):
        self.read_flows(file_name_list[0])
        self.read_analyses(file_name_list[1])
        self.read_buckets(file_name_list[2])
        self.read_orders(file_name_list[3])

    def read_flows(self, file_name_flows='flows.txt'):
        self.file_name_flows = file_name_flows
        self.flows = pd.read_csv(self.file_name_flows, sep=';')
        print("\n\nFLOWS\n")
        print(self.flows)

    def read_analyses(self, file_name_analyses='analyses.txt'):
        self.file_name_analyses = file_name_analyses
        self.analyses = pd.read_csv(self.file_name_analyses, sep=';')
        print("\n\nANALYSES\n")
        print(self.analyses)

    def read_buckets(self, file_name_buckets='buckets.txt'):
        self.file_name_buckets = file_name_buckets
        self.buckets = pd.read_csv(self.file_name_buckets, sep=';')
        print("\n\nBUCKETS\n")
        print(self.buckets)

    def read_orders(self, file_name_orders='orders.txt'):
        self.file_name_orders = file_name_orders
        self.orders = pd.read_csv(self.file_name_orders, sep=';')
        print("\n\nORDERS\n")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(self.orders)

    def show_flows(self):
        return self.flows

    def show_analyses(self):
        return self.analyses

    def show_buckets(self):
        return self.buckets

    def show_orders(self):
        return self.orders



if __name__ == "__main__":

    planing1 = READ()
    # planing1.read_all(['flows.txt', 'analyses.txt', 'buckets.txt', 'orders.txt'])
    planing1.read_all(['flows.txt', 'analyses.txt', 'buckets.txt', 'orders.txt'])
    new_orders = {}
    orders = {}
    steelGrades = {}
    buckets = {}
    wbuckets = []


# // read in data from input files into data structures
    try:

        for i, row in planing1.show_orders().iterrows():
            # print(row['steel grade'], row['maxCharges'], row['combinable'])
            # TODO: Maybe it is better way to create lcost parameter np zip()
            lcost = {'0': float(row['cost0']),'1': float(row['cost1']), '2': float(row['cost2']),'3': float(row['cost3']), '4': float(row['cost4']),'5': float(row['cost5']),'6':float(row['cost6'])}
            o = ORDER(row['steel grade'], row['weight'], row['forbidden'], row['mandatory'], lcost, row['flows'])
            orders[str(i)] = o

        for i, row in planing1.show_orders().iterrows():
            for j, rowj in row.iteritems():
                if j[:4] == 'cost':
                    new_orders[str(i), j[-1]] = rowj

        for i, row in planing1.show_analyses().iterrows():
            # print(row['steel grade'], row['maxCharges'], row['combinable'])
            s = STEELGRADE(row['steel grade'], row['maxCharges'] if row['maxCharges'] > 0 else 1, row['combinable'])
            steelGrades[str(i)] = s

        for i, row in planing1.show_buckets().iterrows():
            # print(row['capacity'])
            b = BUCKET(row['capacity'])
            buckets[str(i)] = b

        wbuckets = sorted(buckets.keys())[:-1]
        # CHECK IF there is no alone steelgrades

        for o in orders.keys():
            flag = False
            for s in steelGrades:
                if orders[o].get_stelgrade() == steelGrades[s].get_steelgrade():
                    flag = True
            if not flag:
                print("Problem with: ", orders[o].get_stelgrade())
                raise Exception


    except IOError:
        print("Something went wrong while reading input.")
    except ValueError:
        print("Something went horribly wrong while reading input.")
    except:
        print("Unsupported error.")
        raise Exception

    """ Create Model Variables """

    model = ConcreteModel(name="(LTSS)")

    # // order variables (fraction planned per order and bucket)
    model.y = Var(orders.keys(), buckets.keys(), within=NonNegativeReals, bounds=(0.0, 1.0), initialize=0.0)

    # // number of charges per steel grade and bucket
    model.x = Var(steelGrades.keys(), buckets.keys(), within=NonNegativeIntegers, initialize=0)

    @model.Constraint(steelGrades.keys(), wbuckets)
    def x_boundary_condition(model, s, b):
        return model.x[s, b] <= buckets[b]()


    # // number of full tundishes per steel grade and bucket
    model.t = Var(steelGrades.keys(), buckets.keys(), within=NonNegativeIntegers, initialize=0)
    @model.Constraint(steelGrades.keys(), wbuckets)
    def t_boundary_condition(model, s, b):
        return model.t[s, b] <= buckets[b]() / steelGrades[s].get_maxCharges()


   # // rest charges (non full tundish) per steel grade and bucket
    model.r = Var(steelGrades.keys(), buckets.keys(), within=NonNegativeIntegers, initialize=0)
    @model.Constraint(steelGrades.keys(), buckets.keys())
    def r_boundary_condition(model, s, b):
        return model.r[s, b] <= steelGrades[s].get_maxCharges()-1


    # // boolean variables for modeling the production cost function
    # model.b0 = Set([range(steelGrades[i].get_maxCharges()) for i in steelGrades.keys()])
    # TODO: can I  add better boundary condition?
    max_global_bucket = 10
    steelGrades_maxCharges_arr = range(max_global_bucket)
    model.b = Var(steelGrades.keys(), buckets.keys(), steelGrades_maxCharges_arr, within = Binary, initialize=1)
    @model.Constraint(steelGrades.keys(), buckets.keys(), steelGrades_maxCharges_arr)
    def b_boundary_condition(model, s, b, m):
        if m <= steelGrades[s].get_maxCharges() - 1:
            return model.b[s,b,m] >= 0 # always true
        else:
            return model.b[s,b,m] == 0 # always false


    # // constraint on b's so that at most 1 is chosen
    @model.Constraint(steelGrades.keys(), buckets.keys())
    def b_limit_condition(model, s, b):
        return sum(model.b[s,b,m] for m in range(steelGrades[s].get_maxCharges())) <= 1


    # # // production cost per steel grade and bucket
    # model.p = Var(steelGrades.keys(), buckets.keys(), within=NonNegativeReals, initialize=0.0)

    # // production cost per steel grade and bucket
    # // model.p = Var(steelGrades.keys(), buckets.keys(), within=NonNegativeReals, initialize = 0.0)


    """ Create Constraints """


    # // order must be planned in a bucket
    @model.Constraint(orders.keys())
    def ored_in_bucket(model, o):
        return sum(model.y[o, b] for b in buckets.keys()) == 1

    # // forbidden/mandatory in first bucket
    @model.Constraint(orders.keys())
    def forbiden_mandatory(model, o):
        if orders[o].get_forbidden():
            return model.y[o,'0'] == 0
        elif orders[o].get_mandatory():
            return model.y[o,'0'] == 1
        else:
            return model.y[o,'0'] <= 1

    # // capacity constraints
    @model.Constraint(wbuckets) # do not check last
    def capacity(model, b):
        return sum(model.x[s, b] for s in steelGrades.keys()) <= buckets[b]()

    # // link planned orders (y) to planned charges (x)
    @model.Constraint(steelGrades.keys(), buckets.keys())
    def link_orders_charges(model, s, b):
        return sum(model.y[o, b] * orders[o].get_weight() if steelGrades[s].get_steelgrade() == orders[o].get_stelgrade() else 0.0 for o in orders.keys()) <= model.x[s, b] * planing1.heatWeight

    # // link boolean variables for production cost (b) to charges (x)

    # ladle conservation law
    # // split x_k, j in t_k, j and r_k, j
    @model.Constraint(steelGrades.keys(), buckets.keys())
    def link_production_charges1(model, s, b):
        return model.t[s,b]*steelGrades[s].get_maxCharges()+model.r[s,b] == model.x[s,b]

    """ objective function """

    # prodCost = planing1.combinableCosts
    # @model.Constraint(steelGrades.keys(), buckets.keys())
    # def model_cost(model, s, b):
    #     prodCost = planing1.combinableCosts if steelGrades[s].get_combinable() else planing1.uncombinableCosts
    #     return model.p[s, b] == model.t[s, b].value * prodCost[int(steelGrades[s].get_maxCharges() - 1)] + prodCost[int(model.r[s, b].value - 1)]



    def obj_rule(model):

        # // tardiness cost
        # TODO: think about how to change int(b)
        MNW = sum(model.y[o, b] * orders[o].get_weight() * new_orders[o, b] for o in orders.keys() for b in buckets.keys())

        for s in steelGrades.keys():
            prodCost = planing1.combinableCosts if steelGrades[s].get_combinable() else planing1.uncombinableCosts
            for b in buckets.keys():

                # // cost of full tundishes
                MNW += model.t[s, b] * prodCost[steelGrades[s].get_maxCharges() - 1]
                # // cost of rest charges
                for m in range(steelGrades[s].get_maxCharges()):
                    MNW += model.b[s, b, m] * prodCost[m -1]

        return MNW


    model.obj = Objective(rule=obj_rule, sense=minimize)


    solver = SolverFactory('glpk') #glpk / ipopt
    solver.options['tmlim'] = 5*60 # seconds
    solver.options['mipgap'] = 0.001 # set relative mip gap tolerance to tol (default = 0.001) (Any number from 0.0 to 1.0
    solver.solve(model, tee=True)

    #model.pprint()
    print("\n\n\n\n ******")
    #model.y.pprint()
    # print("Status = %s" % solver.termination_condition)

    # status = SolverFactory('glpk').solve(model)
    #
    # model.pprint()
    # print("\n\n\n\n ******")
    # model.y.pprint()
    # print("Status = %s" % status.solver.termination_condition)
    '''
    self.solver = pyomo.opt.SolverFactory(SOLVER_NAME)
    if SOLVER_NAME == 'cplex':
        self.solver.options['timelimit'] = TIME_LIMIT
    elif SOLVER_NAME == 'glpk':
        self.solver.options['tmlim'] = TIME_LIMIT
    elif SOLVER_NAME == 'gurobi':
        self.solver.options['TimeLimit'] = TIME_LIMIT
    '''


    """ Print cost"""
    MNW = 0.0
    for s in steelGrades.keys():
        prodCost = planing1.combinableCosts if steelGrades[s].get_combinable() else planing1.uncombinableCosts
        for b in buckets.keys():
            # // cost of full tundishes
            MNW += model.t[s, b].value * prodCost[steelGrades[s].get_maxCharges() - 1]
            # // cost of rest charges
            for m in range(steelGrades[s].get_maxCharges() - 1):
                MNW += model.b[s, b, m].value * prodCost[m -1]




    print("MNW: ", MNW)

    """ Print results"""

    # print("%s = %f" % (model.x, value(model.x)))
    # #print("%s = %f" % (m.h, value(m.h)))
    sbucket = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, }

    print(model.y)
    prev = ('-1', '-1')
    for i1 in model.y:
        if i1[0] == prev[0]:
            print(model.y[i1[0],i1[1]].value, end=' ')
        else:
            print("\nOrder ", i1[0], ": ",model.y[i1[0],i1[1]].value, end=' ')
        prev = i1
        sbucket[i1[1]] += model.y[i1[0],i1[1]].value
    print()

    print("bucket: how many orders")
    for row in sbucket.keys():
        print(f"{row}: {sbucket[row]}")

    print(sbucket)

    # print(model.y)
    # for i1, i2 in model.y:
    #     print (" ",i1, i2, model.y[i1,i2], model.y[i1,i2].value) # doctest: +SKIP

    """ Print ORDERS"""

    print("........")
    print("--------")
    print("________")
    print("")

    total_cost = [0.0 for x in range(7)]
    total_weight = [0.0 for x in range(7)]
    total_mix = []
    for i in range(7):


        print(f"{i} BUCKET")
        print("INDEX | GRADE | WEIGHT | FORBIDDEN | MANDATORY | COST |                 FLOWS")
        for i1 in model.y:
            if i1[1] == str(i) and model.y[i1[0],i1[1]].value == 1.0:
                print(int(i1[0]), end="  ")
                orders[i1[0]].print_fields()
                total_weight[i] += orders[i1[0]].get_weight()
                total_cost[int(i1[1])] += orders[i1[0]].get_cost(i1[1])
        print()
        total_mix.append(total_weight[i]*total_cost[i])
        print(f"TOTAL WEIGHT: {total_weight[i]} TOTAL COST: {total_cost[i]} WEIGHTxCOST: {total_weight[i]*total_cost[i]}")
        print()
        print()
    print("SUM COST [COST PER BUCKET]: ", sum(total_mix), total_mix)
    print("SUM WEIGHT [WEIGHT PER BUCKET]: ", sum(total_weight), total_weight)
    total_ladle = [x/planing1.heatWeight for x in total_weight]
    print("SUM LADLE [LADLE PER BUCKET]: ", sum(total_ladle), total_ladle)


    """ Print STEELGRADE"""

    print("........")
    print("--------")
    print("________")
    print("")

    print(model.x)
    prev = ('-1', '-1')
    for i1 in model.x:
        if i1[0] == prev[0]:
            print(model.x[i1[0],i1[1]].value, end=' ')
        else:
            print("\nGrade ", steelGrades[i1[0]].get_steelgrade(), ": ",model.x[i1[0],i1[1]].value, end=' ')
        prev = i1
        sbucket[i1[1]] += model.x[i1[0],i1[1]].value
    print()


    """ Print TUNDISCHES"""

    print("........")
    print("--------")
    print("________")
    print("")

    print(model.t)
    prev = ('-1', '-1')
    for i1 in model.t:
        if i1[0] == prev[0]:
            print(model.t[i1[0],i1[1]].value, end=' ')
        else:
            print("\nGrade ", steelGrades[i1[0]].get_steelgrade(), ": ",model.t[i1[0],i1[1]].value, end=' ')
        prev = i1
        sbucket[i1[1]] += model.t[i1[0],i1[1]].value
    print()


    """ Print REST"""

    print("........")
    print("--------")
    print("________")
    print("")

    print(model.r)
    prev = ('-1', '-1')
    for i1 in model.r:
        if i1[0] == prev[0]:
            print(model.r[i1[0],i1[1]].value, end=' ')
        else:
            print("\nGrade ", steelGrades[i1[0]].get_steelgrade(), ": ",model.r[i1[0],i1[1]].value, end=' ')
        prev = i1
        sbucket[i1[1]] += model.r[i1[0],i1[1]].value
    print()

###################################################
    stop = timeit.default_timer()
    print(f'Time: {(stop - start)} sec')
    # MAIN END