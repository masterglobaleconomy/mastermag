
import math
from functools import reduce
from operator import mul

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


"""

dwie klasy:

Trajectory - zawiera metody do rozwiązania kilku lańcuchów w celach demonstracyjnych
TrajectoryAuto - jest uogólnieniem rozwiązywanego łańcucha (można wprowadzić n elementów) 
! nie rozwiązuje łańcóchów cyklicznych czyli np lam[xi] = lam[xi+1]  - wszystkie elementy wektora muszą byc różne


"""

class TrajectoryAuto():

    def __init__(self, b: list, lam: list):

        self.b = b
        self.lam = lam
        self.t = 0

        # assert

    @property
    def B(self):
        return 1 # reduce(lambda x, y: mul(x, y), self.b)


    def alpha(self, i, n):

        alpha_i = 1

        for j in range(n):
            if j != i:
                alpha_i *= self.lam[j] / (self.lam[j] - self.lam[i])

        return alpha_i

    def transition(self, n, t):

        self.t = t
        s = 0
        for i in range(n):
            s += self.lam[i] * self.alpha(i, n) * math.exp(-self.lam[i] * self.t)
        # print(self.B, s, self.lam, n)
        return self.B * s / self.lam[n-1]

    def passage(self, n): # is defined assuming that the next nuclide is artifically stable

        s = 0
        for i in range(n):
            s += self.lam[i] * self.alpha(i, n) * math.exp(-self.lam[i] * self.t)

        s += 0
        return self.B * s / self.lam[n-1]

class Trajectory():


    def __init__(self, b: list, lam: list, t: float):
        self.b = b
        self.lam = lam
        self.t = t

    """
    First Trajecotry (Remaining)
    Ex: A remaining
    """
    @property
    def  transition0(self):
        return math.exp(-self.lam[0] * self.t)

    @property
    def passage0(self):
        return 1 - math.exp(-self.lam[0] * self.t)


    """
    Second trajectory
    Ex: A -> B with bAB
    """

    @property
    def transition1(self):
        return self.b[0] * (
                              ((self.lam[0] / (self.lam[1] - self.lam[0])) * math.exp(-self.lam[0] * self.t))
                            - (math.exp(-self.lam[1] * self.t))
        )

    @property
    def passage1(self):
        return self.b[0] * (
                            1
                            - ((self.lam[1] / (self.lam[1] - self.lam[0])) * math.exp(-self.lam[0] * self.t))
                            - ((self.lam[0] / (self.lam[0] - self.lam[1])) * math.exp(-self.lam[1] * self.t))
        )



    """
    Third trajectory
    Ex: A -> B -> C with bAB and bBC
    """

    @property
    def transition2(self):
        if self.lam[0] == self.lam[2]:
            return self.transition2c
        else:
            return self.b[0] * self.b[1] * (
                              ((self.lam[0] * self.lam[1] / ((self.lam[1] - self.lam[0]) * (self.lam[2] - self.lam[0]))) * math.exp(-self.lam[0] * self.t))
                            + ((self.lam[0] * self.lam[1] / ((self.lam[0] - self.lam[1]) * (self.lam[2] - self.lam[1]))) * math.exp(-self.lam[1] * self.t))
                            + ((self.lam[0] * self.lam[1] / ((self.lam[0] - self.lam[2]) * (self.lam[1] - self.lam[2]))) * math.exp(-self.lam[2] * self.t))
            )

    @property
    def passage2(self):
        if self.lam[0] == self.lam[2]:
            return self.passage2c
        else:
            return self.b[0] * self.b[1] * (
                            1
                            - ((self.lam[1] * self.lam[2] / ((self.lam[1] - self.lam[0]) * (self.lam[2] - self.lam[0]))) * math.exp(-self.lam[0] * self.t))
                            - ((self.lam[0] * self.lam[2] / ((self.lam[0] - self.lam[1]) * (self.lam[2] - self.lam[1]))) * math.exp(-self.lam[1] * self.t))
                            - ((self.lam[0] * self.lam[1] / ((self.lam[0] - self.lam[2]) * (self.lam[1] - self.lam[2]))) * math.exp(-self.lam[2] * self.t))
            )

    """
    Cyclic 3 element
    Ex: A -> B -> A with bAB / bBA
    lam[0] = lam[2]
    """

    @property
    def transition2c(self):
        return self.b[0] * self.b[1] * (
                      ((self.lam[1] / (self.lam[1] - self.lam[0])) * math.exp(-self.lam[0] * self.t))
                    * (self.lam[0] / (self.lam[0] - self.lam[1]) + self.lam[0] * self.t)
                    + ( (self.lam[1] / self.lam[0]) * math.pow(self.lam[0] / (self.lam[0] - self.lam[1]), 2) * math.exp(-self.lam[1] * self.t))
        )

    @property
    def passage2c(self):
        return self.b[0] * self.b[1] * (
                    1
                    - ( (self.lam[1] / (self.lam[1] - self.lam[0]) * math.exp(-self.lam[0] * self.t)) * (1 + self.lam[0] / (self.lam[0] - self.lam[1]) + self.lam[0] * self.t) )
                    - ( (math.pow(self.lam[0] / (self.lam[0] - self.lam[1]), 2)) * math.exp(-self.lam[1] * self.t))
        )


if __name__ == "__main__":


    #############################
    # One Linear Chain
    T = TrajectoryAuto([1], [1, 1.2, 0.8, 0.00001])  # pierwszy wektor to Branchink, potrzebny przy analizie większej ilości łańcuchów
    # Drugi wektor to wskaźniki wydajności procesu, przejścia (produkcji, destrukcji - jak z minusem)

    X1 = np.linspace(0, 10, 500)
    Y1 = [T.transition(1, t) for t in X1]
    Y2 = [T.transition(2, t) for t in X1]
    Y3 = [T.transition(3, t) for t in X1]
    Y4 = [T.transition(4, t) for t in X1]
    # P2 = [T.transition(4, t) for t in X1]
    # R = [y1+y2+y3+p2 for y1, y2, y3, p2 in zip(Y1, Y2, Y3,P2)]
    # R1 = [y2 + y3 + p2 for y2, y3, p2 in zip(Y2, Y3, P2)]
    # A = [y1 + y3  for y1, y3, in zip(Y1, Y3)]

    a = plt.scatter(X1, Y1, s=0.5)# , alpha=0.5)
    b = plt.scatter(X1, Y2, s=0.5)
    c = plt.scatter(X1, Y3, s=0.5)
    d = plt.scatter(X1, Y4, s=0.5)
    # plt.scatter(X1, P2, s=0.5)
    # plt.scatter(X1, A, s=0.5)
    # plt.scatter(X1, R, s=0.5)
    # plt.scatter(X1, R1, s=0.5)
    plt.legend((a, b, c, d), ('a', 'b', 'c', 'd'))
    plt.show()

    dataframe = pd.DataFrame({'Y1': Y1, 'Y2': Y2, 'Y3': Y3, 'Y4': Y4})

    # print(dataframe)
    sns.pairplot(dataframe,  diag_kind="hist") #

    plt.show()