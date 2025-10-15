import numpy as np
from scipy.special import erf


# ------  Activation Functions & Derivatives  ------ #

def RELU(x):
    return np.maximum(x, 0)

def dRELU(x):
    return np.where(x > 0, 1, 0)


def ABS(x):
    return np.abs(x)

def dABS(x):
    return np.where(x > 0, 1, np.where(x < 0, -1, 0))


def GELU(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

def dGELU(x):
    # Derivative: 0.5 * (1 + erf(x/sqrt(2))) + (x / sqrt(2π)) * exp(-x²/2)
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0))) + (x * np.exp(-x**2 / 2.0)) / np.sqrt(2.0 * np.pi)


def LINEAR(x):
    return x

def dLINEAR(x):
    return np.ones_like(x)


def BINARY(x):
    return np.where(x >= 0, 1, 0)

def dBINARY(x):
    # Derivative is zero everywhere except undefined at x=0
    return np.zeros_like(x)


def SIGMOID(x):
    return 1 / (1 + np.exp(-x))

def dSIGMOID(x):
    s = SIGMOID(x)
    return s * (1 - s)


def TANH(x):
    return np.tanh(x)

def dTANH(x):
    t = np.tanh(x)
    return 1 - t**2


def SOFTSIGN(x):
    return x / (1 + np.abs(x))

def dSOFTSIGN(x):
    return 1 / (1 + np.abs(x))**2


def SOFTPLUS(x):
    return np.log1p(np.exp(x))

def dSOFTPLUS(x):
    # Derivative of softplus is sigmoid
    return 1 / (1 + np.exp(-x))


def LEAKY_RELU(x):
    return np.where(x > 0, x, 0.01 * x)

def dLEAKY_RELU(x):
    return np.where(x > 0, 1, 0.01)


def SILU(x):
    # Also known as Swish: x * sigmoid(x)
    return x / (1 + np.exp(-x))

def dSILU(x):
    s = 1 / (1 + np.exp(-x))
    return s + x * s * (1 - s)


def ELISH(x):
    # x < 0: (exp(x) - 1)/(1 + exp(-x))
    # x >= 0: x/(1 + exp(-x))
    return np.where(x < 0,
                    (np.exp(x) - 1) / (1 + np.exp(-x)),
                    x / (1 + np.exp(-x)))

def dELISH(x):
    # Piecewise derivative
    pos = (1 / (1 + np.exp(-x))) + (x * np.exp(-x)) / (1 + np.exp(-x))**2
    neg = (np.exp(x) * (1 + np.exp(-x)) - (np.exp(x) - 1) * np.exp(-x)) / (1 + np.exp(-x))**2
    return np.where(x < 0, neg, pos)


def GAUSSIAN(x):
    return np.exp(-x**2)

def dGAUSSIAN(x):
    return -2 * x * np.exp(-x**2)


def SINUSOIDAL(x):
    return np.sin(x)

def dSINUSOIDAL(x):
    return np.cos(x)