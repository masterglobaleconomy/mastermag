""" 
Chebyshev Rational Approximation Method module

Implements two different forms of CRAM for use in opendeplete.
"""

import numpy as np
import scipy.sparse as sp
import torch
from matplotlib import pyplot as plt
import pandas as pd

class CRAM():

    def __init__(self):
        pass

    def __call__(self, A, n0, dt, method = 1):
        if method == 1:
            return self.CRAM16(A, n0, dt)
        else:
            return self.CRAM48(A, n0, dt)

    def cram_wrapper(self, chain, n0, rates, dt):
        """Wraps depletion matrix creation / CRAM solve for multiprocess execution

        Parameters
        ----------
        chain : DepletionChain
            Depletion chain used to construct the burnup matrix
        n0 : numpy.array
            Vector to operate a matrix exponent on.
        rates : numpy.ndarray
            2D array indexed by nuclide then by cell.
        dt : float
            Time to integrate to.

        Returns
        -------
        numpy.array
            Results of the matrix exponent.
        """
        A = chain.form_matrix(rates)
        return self.CRAM48(A, n0, dt)

    def CRAM16(self, A, n0, dt):
        """ Chebyshev Rational Approximation Method, order 16 (PyTorch version)

        This implementation uses PyTorch dense tensors and torch.linalg.solve.
        It converts sparse/dense numpy/ scipy inputs to dense torch tensors,
        performs the IPF CRAM16 iterations and returns a numpy array (real).
        """


        # coefficients (same values as original)
        alpha_np = np.array([+2.124853710495224e-16,
                                +5.464930576870210e+3 - 3.797983575308356e+4j,
                                +9.045112476907548e+1 - 1.115537522430261e+3j,
                                +2.344818070467641e+2 - 4.228020157070496e+2j,
                                +9.453304067358312e+1 - 2.951294291446048e+2j,
                                +7.283792954673409e+2 - 1.205646080220011e+5j,
                                +3.648229059594851e+1 - 1.155509621409682e+2j,
                                +2.547321630156819e+1 - 2.639500283021502e+1j,
                                +2.394538338734709e+1 - 5.650522971778156e+0j],
                            dtype=np.complex128)
        theta_np = np.array([+0.0,
                                +3.509103608414918 + 8.436198985884374j,
                                +5.948152268951177 + 3.587457362018322j,
                                -5.264971343442647 + 16.22022147316793j,
                                +1.419375897185666 + 10.92536348449672j,
                                +6.416177699099435 + 1.194122393370139j,
                                +4.993174737717997 + 5.996881713603942j,
                                -1.413928462488886 + 13.49772569889275j,
                                -10.84391707869699 + 19.27744616718165j],
                            dtype=np.complex128)

        # Ensure A is dense numpy array
        if sp.issparse(A):
            A_arr = A.toarray()
        else:
            A_arr = np.asarray(A)

        # Convert to torch complex tensors (double precision complex)
        device = torch.device("cpu")
        A_t = torch.tensor(A_arr, dtype=torch.complex128, device=device)
        n = A_t.shape[0]
        I = torch.eye(n, dtype=torch.complex128, device=device)

        alpha = torch.tensor(alpha_np, dtype=torch.complex128, device=device)
        theta = torch.tensor(theta_np, dtype=torch.complex128, device=device)

        alpha0 = 2.124853710495224e-16
        k = 8

        # prepare y as complex torch tensor
        if isinstance(n0, list):
            y = torch.tensor(n0, dtype=torch.complex128, device=device)
        else:
            # accept numpy array or torch tensor
            if isinstance(n0, torch.Tensor):
                y = n0.to(device=device, dtype=torch.complex128)
            else:
                y = torch.tensor(np.asarray(n0), dtype=torch.complex128, device=device)

        # CRAM IPF loop (note original loop uses indices 1..k)
        for l in range(1, k + 1):
            M = A_t * float(dt) - theta[l] * I
            # solve M x = y
            x = torch.linalg.solve(M, y)
            y = 2.0 * torch.real(alpha[l] * x) + y

        y = y * float(alpha0)

        # return real numpy array (matches original behaviour)
        return y.real.cpu().numpy()

    def CRAM48(self, A, n0, dt):
        """Chebyshev Rational Approximation Method, order 48 (PyTorch version)"""
        
        # coefficients
        theta_r = np.array([-4.465731934165702e+1, -5.284616241568964e+0,
                            -8.867715667624458e+0, +3.493013124279215e+0,
                            +1.564102508858634e+1, +1.742097597385893e+1,
                            -2.834466755180654e+1, +1.661569367939544e+1,
                            +8.011836167974721e+0, -2.056267541998229e+0,
                            +1.449208170441839e+1, +1.853807176907916e+1,
                            +9.932562704505182e+0, -2.244223871767187e+1,
                            +8.590014121680897e-1, -1.286192925744479e+1,
                            +1.164596909542055e+1, +1.806076684783089e+1,
                            +5.870672154659249e+0, -3.542938819659747e+1,
                            +1.901323489060250e+1, +1.885508331552577e+1,
                            -1.734689708174982e+1, +1.316284237125190e+1])
        theta_i = np.array([+6.233225190695437e+1, +4.057499381311059e+1,
                            +4.325515754166724e+1, +3.281615453173585e+1,
                            +1.558061616372237e+1, +1.076629305714420e+1,
                            +5.492841024648724e+1, +1.316994930024688e+1,
                            +2.780232111309410e+1, +3.794824788914354e+1,
                            +1.799988210051809e+1, +5.974332563100539e+0,
                            +2.532823409972962e+1, +5.179633600312162e+1,
                            +3.536456194294350e+1, +4.600304902833652e+1,
                            +2.287153304140217e+1, +8.368200580099821e+0,
                            +3.029700159040121e+1, +5.834381701800013e+1,
                            +1.194282058271408e+0, +3.583428564427879e+0,
                            +4.883941101108207e+1, +2.042951874827759e+1])
        theta_np = np.array(theta_r + theta_i * 1j, dtype=np.complex128)

        alpha_r = np.array([+6.387380733878774e+2, +1.909896179065730e+2,
                            +4.236195226571914e+2, +4.645770595258726e+2,
                            +7.765163276752433e+2, +1.907115136768522e+3,
                            +2.909892685603256e+3, +1.944772206620450e+2,
                            +1.382799786972332e+5, +5.628442079602433e+3,
                            +2.151681283794220e+2, +1.324720240514420e+3,
                            +1.617548476343347e+4, +1.112729040439685e+2,
                            +1.074624783191125e+2, +8.835727765158191e+1,
                            +9.354078136054179e+1, +9.418142823531573e+1,
                            +1.040012390717851e+2, +6.861882624343235e+1,
                            +8.766654491283722e+1, +1.056007619389650e+2,
                            +7.738987569039419e+1, +1.041366366475571e+2])
        alpha_i = np.array([-6.743912502859256e+2, -3.973203432721332e+2,
                            -2.041233768918671e+3, -1.652917287299683e+3,
                            -1.783617639907328e+4, -5.887068595142284e+4,
                            -9.953255345514560e+3, -1.427131226068449e+3,
                            -3.256885197214938e+6, -2.924284515884309e+4,
                            -1.121774011188224e+3, -6.370088443140973e+4,
                            -1.008798413156542e+6, -8.837109731680418e+1,
                            -1.457246116408180e+2, -6.388286188419360e+1,
                            -2.195424319460237e+2, -6.719055740098035e+2,
                            -1.693747595553868e+2, -1.177598523430493e+1,
                            -4.596464999363902e+3, -1.738294585524067e+3,
                            -4.311715386228984e+1, -2.777743732451969e+2])
        alpha_np = np.array(alpha_r + alpha_i * 1j, dtype=np.complex128)

        # Ensure A is dense numpy array
        if sp.issparse(A):
            A_arr = A.toarray()
        else:
            A_arr = np.asarray(A)

        # Convert to torch tensors
        device = torch.device("cpu")
        A_t = torch.tensor(A_arr, dtype=torch.complex128, device=device)
        n = A_t.shape[0]
        I = torch.eye(n, dtype=torch.complex128, device=device)

        alpha = torch.tensor(alpha_np, dtype=torch.complex128, device=device)
        theta = torch.tensor(theta_np, dtype=torch.complex128, device=device)

        alpha0 = 2.258038182743983e-47
        k = 24

        # prepare y as complex torch tensor
        if isinstance(n0, list):
            y = torch.tensor(n0, dtype=torch.complex128, device=device)
        else:
            if isinstance(n0, torch.Tensor):
                y = n0.to(device=device, dtype=torch.complex128)
            else:
                y = torch.tensor(np.asarray(n0), dtype=torch.complex128, device=device)

        # CRAM IPF loop
        for l in range(k):
            M = A_t * float(dt) - theta[l] * I
            x = torch.linalg.solve(M, y)
            y = 2.0 * torch.real(alpha[l] * x) + y

        y = y * float(alpha0)

        return y.real.cpu().numpy()


def test_cram() -> None:

    A = np.matrix([[-1, 0 , 0, 0], [1, -1.2, 0, 0], [0, 1.2, -0.8, 0], [0, 0, 0.8, 0]])  # input matrix
    B = np.array([0.0, 0.0, 0.0, 0.0]) # diagonal matrix addin increase (decrease)  dA/dt = s*A

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

    a = plt.scatter(X, Y1, s=0.5)
    b = plt.scatter(X, Y2, s=0.5)
    c = plt.scatter(X, Y3, s=0.5)
    d = plt.scatter(X, Y4, s=0.5)
    plt.legend((a, b , c, d), ('a', 'b', 'c', 'd'))
    plt.show()

if __name__ == "__main__":

    import random
    # test_cram()

    dt = 1
    data = pd.read_csv("src/master_equation/test_data.csv")

    def simple_data_loader(data: pd.DataFrame, window: int = 10):
        """Generator yielding (index, y0, y1) pairs where y1 is window steps after y0."""
        n_rows = len(data)
        if n_rows <= window:
            return
        for _i in range(0, n_rows - window):
            idx = random.randint(0, n_rows - window - 1)
            y0 = data.iloc[idx, 1:5].to_numpy()
            y1 = data.iloc[idx + window, 1:5].to_numpy()
            yield idx, y0, y1

    # # example: iterate with enumeration
    # for epoch in range(2):
    #     print(f"Epoch {epoch}")
    #     for idx, (i, y0, y1) in enumerate(simple_data_loader(data)):

    #         print(idx, i, y0, y1)
    #         if idx >= 12:  # stop after a few examples
    #             break

    cram = CRAM()
    dt = 10*10/500 #np.linspace(0, 10, 500)

    # for i in np.linspace(0, 100, 11): # 10 steps
    i = 0
    w = torch.tensor(0.0, requires_grad=True)
    for epoch in range(100):
        y0 = data.iloc[int(i), 2:].to_numpy()
        y1 = data.iloc[int(i + 10), 2:].to_numpy()
        # A = np.matrix([[-1, 0 , 0, 0], [1, -1.2, 0, 0], [0, 1.2, -0.8, 0], [0, 0, 0.8, 0]])  # input matrix 
        A = np.matrix([[-1, 0 , 0, 0], [1, -1.2, 0, 0], [0, w, -0.8, 0], [0, 0, 0.8, 0]])  # input matrix 

        y_pred = cram(A, y0, dt)

        print(f"y_pred: {y_pred}  expected: {y1}")
        # print(f"  y0:    {y0}")
        # print(f"  y1:    {y1}")
        # print(f"  y_pred:{y_pred}")
