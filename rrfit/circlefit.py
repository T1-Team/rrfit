"""
Algebraic circle fit for analyzing complex S21 data of superconducting resonators
Adapted from Probst et al 2015.
"""

from itertools import product
import numpy as np
from scipy import optimize

from rrfit.models import S21CenteredPhaseModel


def fit_circle(s21):
    """
    we assume our n data points approximately form a circle A*zi + B*xi + C*yi + D ~ 0
    we want to minimize the function: sum of (A*zi + B*xi + C*yi + D)^2 from i=1 to i=n
    this function can be written as P.t * M * P - m * (P.t * B * P - 1)
    where vector P is [A  B  C  D].t, where .t indicates the transpose,
    M is the matrix of moments,
    B is the matrix that represents our constraint B^2 + C^2 - 4*A*D = 1,
    and m is the lagrange multiplier

    s21: one dimensional np array of complex numbers
    return tuple (radius, center), center is complex number xc + 1j * yc
    """
    # rescale raw data to get numbers of a similar magnitude
    # 'translator' is the translating summand and 'scalar' is the scaling factor
    scaleddata, translator, scalar = rescale(s21)

    # calculate the matrix of moments M based on the scaled data
    M = get_moments(scaleddata)

    # calculate the coefficients of charpoly from M
    k0, k1, k2, k3, k4 = get_charpoly_coeffs(M)

    # calculate the eigenvalue l that solves characteristic polynomial det(M - mB) = 0
    l = solve_charpoly(k0, k1, k2, k3, k4)

    # adjust a subset of the moments using the computed eigenvalue l
    M[3][0] = M[0][3] = M[3][0] + 2 * l
    M[1][1], M[2][2] = M[1][1] - l, M[2][2] - l

    # find the eigenvector P = [A B C D].t corresponding to l
    P = get_circle_coeffs(M)
    A, B, C, D = P.tolist()

    # extract radius and center coordinates of the fitted circle from P
    # re-scale and re-translate the radius and center before returning them
    radius = (np.sqrt(B**2 + C**2 - 4 * A * D) / (2 * abs(A))) * abs(scalar)
    center = ((-B / (2 * A)) + 1j * (-C / (2 * A))) * scalar + translator

    return radius, center


def rescale(data):
    """
    do min-max scaling of input data
    note that we don't alter the shape of the data, just its magnitude
    this rescaling allows our numerical solvers to converge better

    data: one dimensional np array of complex numbers
    """
    xmin, xmax = min(data.real), max(data.real)
    ymin, ymax = min(data.imag), max(data.imag)
    translator, scalar = xmin + 1j * ymin, (xmax + 1j * ymax) - (xmin + 1j * ymin)
    scaleddata = (data - translator) / scalar
    return scaleddata, translator, scalar


def get_moments(data):
    """
    returns M, the numeric 4x4 matrix of moments from received data
    M = [[Mzz   Mxz  Myz  Mz],
        [ Mxz   Mxx  Mxy  Mx],
        [ Myz   Mxy  Myy  My],
        [  Mz    Mx   My   n]]
    where Mij = sum from 1 to n of i*j where i and j are 1D arrays of real numbers
    and n is the number of data points
    """
    n = float(data.shape[0])  # number of data points
    xi, yi = data.real, data.imag
    zi = (xi * xi) + (yi * yi)

    def moment(x, y, n):
        """
        calculates the moment if both x and y are np arrays, else returns n as received.
        """
        try:
            return (x * y).sum()  # typical case
        except AttributeError:  # edge case e.g. if x == 1 and y == 1
            return n

    moments = [moment(*pair, n) for pair in product([zi, xi, yi, 1], repeat=2)]
    return np.array(moments).reshape(4, 4)


def solve_charpoly(k0, k1, k2, k3, k4):
    """
    k0, k1, k2, k3, k4: coefficients of the characteristics polynomial
    l is the smallest non-negative eigenvalue solved for using Newton-Raphson method. As the charpoly is decreasing and concave up between 0 and l, we are guaranteed to converge on l with an initial value of 0.
    """

    # define the charpoly and its derivative
    def charpoly(l):
        return k0 + k1 * l + k2 * l**2 + k3 * l**3 + k4 * l**4

    def dcharpoly(l):
        return k1 + 2 * k2 * l + 3 * k3 * l**2 + 4 * k4 * l**3

    l = optimize.newton(charpoly, 0.0, dcharpoly)
    return l


def get_charpoly_coeffs(M):
    """
    We have pre-computed expressions for the 5 coefficients of the characteristic polynomial det(M - lB) = 0 for efficiency using this code:

    from sympy import Matrix, MatrixSymbol, symbols
    M = MatrixSymbol("M", 4, 4)
    B = Matrix([[0.0, 0.0, 0.0, -2.0], [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0], [-2.0, 0.0, 0.0, 0.0]])
    k = symbols("k")
    print (Matrix(M - k * B).det())

    Note that the printed expression has been rearranged to prevent floating point math errors due to rounding and/or catastrophic cancellation. The values of k0 and k1 in particular tend to be numerically unstable if computed using the naive simplified expression that sympy provides.
    """
    # do a change of variables to make the coefficient expressions neater
    a, b, c, d, e = M[0][0], M[0][1], M[0][2], M[0][3], M[1][1]
    f, g, h, i, j = M[1][2], M[1][3], M[2][2], M[2][3], M[3][3]
    k0 = (
        (b**2 * (i**2 - h * j) + c**2 * (g**2 - e * j))
        + (f**2 * (d**2 - a * j) + e * h * (a * j - d**2))
        + (2 * b * f * (c * j - d * i) + 2 * c * d * (e * i - f * g))
        + (2 * b * g * (d * h - c * i) + a * (2 * f * g * i - e * i**2 - g**2 * h))
    )
    k1 = (
        a * ((g**2 - e * j) + (i**2 - h * j))
        + d * ((d * (e + h) - (b * g + c * i)) + 4 * (f**2 - e * h))
        + c * ((c * j - i * d) + 4 * (e * i - g * f))
        + b * ((b * j - d * g) + 4 * (g * h - f * i))
    )
    k2 = (a * j - d**2) + 4 * ((e * d - c * i) + (h * d - b * g) + (f**2 - h * e))
    k3 = 4 * (e + h - d)
    k4 = -4.0
    return k0, k1, k2, k3, k4


def get_circle_coeffs(M):
    """
    find the eigenvalues and eigenvectors of the Hermitian matrix M
    the eigenvector P we seek is the one associated with the smallest eigenvalue
    """
    w, v = np.linalg.eigh(M)
    return v[:, np.argmin(w)]


def fit_background(s21, f, discont=1.5 * np.pi, windowx=0.25):
    """ """
    radius, center = fit_circle(s21)

    model = S21CenteredPhaseModel()
    centered_phase = model.center_phase(s21 - center, discont=discont, windowx=windowx)
    result = model.fit(centered_phase, f)

    theta = result.best_values["theta"]
    beta = ((theta + np.pi) % (2 * np.pi)) - np.pi
    rp = center + radius * np.cos(beta) + 1j * radius * np.sin(beta)
    orp = center + (center - rp)

    return orp