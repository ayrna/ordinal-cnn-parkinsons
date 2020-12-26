"""
Method for deciding the parameters alpha and beta of a beta distribution, by specifying two
quantile constraints, as described in the following publication:
Van Dorp, J. René, and Thomas A. Mazzuchi. 2000.
“Solving For the Parameters of a Beta a Distribution under Two Quantile Constraints.”
Journal of Statistical Computation and Simulation 67 (2): 189–201.
https://doi.org/10.1080/00949650008812041.
"""

import scipy.stats as st


# Reparametrization of the beta distribution as used in (Van Dorp and Mazzuchi 2000)
beta_d = lambda alpha, beta: st.beta(alpha*beta, beta*(1-alpha))
default_delta = 1e-5

def bisect1(alpha, beta, q, delta=default_delta):
    """
    Let X ~ Beta(beta*alpha, beta*(1-alpha))
    This function solves for the q-th quantile x_q of X
    """
    B = beta_d(alpha, beta).cdf
    xq = 0
    qm = 0
    
    d, e = 0, 1
    while abs(qm - q) >= delta:
        xq = (d + e) / 2
        qm = B(xq)
        
        if qm <= q:
            d = xq
        else:
            e = xq
    
    return xq


def bisect2(xq, beta, q, delta=default_delta):
    """
    Let X ~ Beta(beta*alpha, beta*(1-alpha))
    This function solves for the parameter alpha^o
    that satisfies Pr{X <= x_q} = q
    """
    alpha = 0
    qn = 0
    d, e = 0, 1
    while abs(qn - q) >= delta:
        alpha = (d + e) / 2
        qn = beta_d(alpha, beta).cdf(xq)
        if qn <= q:
            e = alpha
        else:
            d = alpha
    return alpha


def bisect3(xql, xqu, ql, qu, delta=default_delta):
    """
    Let X ~ Beta(beta*alpha, beta*(1-alpha))
    This function solves for both parameters alpha* and beta*
    that satisfies Pr{X <= x_ql} = ql and Pr{X <= x_qu} = qu
    """
    if xql > xqu:
        xql, xqu = xqu, xql
        ql, qu = qu, ql
    
    beta_k = 1
    xql_1k = 0
    while True:
        alpha = bisect2(xqu, beta_k, qu, delta)
        xql_1k = bisect1(alpha, beta_k, ql, delta)
        if xql_1k < xql:
            beta_k = 2 * beta_k
        else:
            a, b = 0, beta_k
            break
    
    xql_k = 0
    alpha, beta = 0, 0
    while (abs(xql_k - xql) >= delta):
        beta  = (a+b) / 2
        alpha = bisect2(xqu, beta, qu, delta)
        xql_k = bisect1(alpha, beta, ql, delta)
        if xql_k < xql:
            a = beta
        else:
            b = beta
    a = alpha * beta
    b = beta * (1-alpha)
    return a, b
