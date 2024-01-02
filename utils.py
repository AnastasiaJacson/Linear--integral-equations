import matplotlib.pyplot as plt
import numpy as np
import math

R = 2 # радіус кола
def xi(t):
    return np.array([R*math.cos(t), R*math.sin(t)])

def xi_der(t):
    return np.array([-R*math.sin(t), R*math.cos(t)])

def xi_der2(t):
    return np.array([-R*math.cos(t), -R*math.sin(t)])

def gradPhi(x, y):
    return (x - y) / (2 * np.pi * np.linalg.norm(x - y)**2)

def nu(t):
    # обчислюємо одиничний вектор нормалі 
    # до кола радіуса R в точці xi
    der = xi_der(t)
    mod = np.linalg.norm(der)
    nu = np.array([der[1] / mod, -der[0] / mod])
    return nu

def K(t, tau):
    # if t == tau:
    #     return - xi_der2(t) @ nu(t) / (2*np.pi*np.linalg.norm(xi_der(t))**2)
    #     # return xi_der2(t) @ nu(t) / (2*np.pi*np.linalg.norm(xi_der(t))**2)
    # else:
    #     return (xi(t) - xi(tau)) @ nu(tau) / (np.pi*np.linalg.norm(xi(t) - xi(tau))**2)if t == tau:
    if t == tau:
        return xi_der2(t) @ nu(t) / (2*np.pi*np.linalg.norm(xi_der(t)))
        # return xi_der2(t) @ nu(t) / (2*np.pi*np.linalg.norm(xi_der(t))**2)
    else:
        return np.linalg.norm(xi_der(tau)) * (xi(t) - xi(tau)) @ nu(tau) / (np.pi*np.linalg.norm(xi(t) - xi(tau))**2)
    
def psi_for_points(t_coloc: np.ndarray):
    n = len(t_coloc)
    h = t_coloc[1] - t_coloc[0]
    def psi(j: int, t: float):
        if j > 0 and t >= t_coloc[j-1] and t <= t_coloc[j]:
            return (t - t_coloc[j-1]) / h
        elif j < n - 1 and t >= t_coloc[j] and t <= t_coloc[j+1]:
            return (t_coloc[j+1] - t) / h
        else:
            return 0

    return psi

def gauss_quadrature(a: float, b: float, m: int, f):
    omega = [0.55555556, 0.88888889, 0.55555556]
    t = [-0.77459667, 0, 0.77459667]
    n = 3
    h = (b - a) / m
    I = 0
    for c in [a + j * h for j in range(m)]:
        I += sum([omega[i] * f((2*c + h) / 2 + h * t[i] / 2) for i in range(n)])
    
    return h / 2 * I

def find_phi(n: int, m: int, f, silent=True):
    t_coloc = np.linspace(0, 2*np.pi, n+1)
    psi = psi_for_points(t_coloc)

    A = np.zeros((n+1, n+1))
    B = -2 * np.vectorize(f)(t_coloc)

    for i in range(n+1):
        for j in range(n+1):
            #tau or t_coloc_i in psi
            qd = gauss_quadrature(a=0, b=2*np.pi, m=m, f=lambda tau: K(t_coloc[i], tau)*psi(j, tau))
            if not silent:
                print('qd =', qd)
            A[i,j] = psi(j, t_coloc[i]) - qd

    C = np.linalg.solve(A, B)
    if not silent:
        print('A =', A)
        print('B =', B)
        print('C =', C)

    phi = lambda x: sum([C[j] * psi(j, x) for j in range(n+1)])

    return phi

def find_u(n: int, m_phi: int, m_u: int, f, silent=True):
    phi = find_phi(f=f, n=n, m=m_phi, silent=silent)
    u = lambda x: gauss_quadrature(a=0, b=2*np.pi, m=m_u, f=lambda tau: phi(tau) * (gradPhi(x, xi(tau)) @ nu(tau)) * np.linalg.norm(xi_der(tau)))

    return u, phi
