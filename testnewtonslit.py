import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

e = 0.2
a0 = - np.pi / 2 #alpha
a1 = - np.pi / 4
a2 = 0
a3 = np.pi / 4
h = 0.001 #step size
v1 = 0
v2 = 1
v3 = 0
z0 = 5j*e
hw = 1 #height of wall

def mu(alpha):
    return e * v2 * np.exp(2j * alpha)

def lam(alpha):
    return -(e**2) * v1  * np.exp(1j * alpha)

def zeta(z):
    return -(z**2)/(z**2+2*hw**2) + 2j*hw*np.sqrt(z+1j*hw)*np.sqrt(z-1j*hw)/(z**2+2*hw**2)

def dz1(z):
    return 1j*hw*(2)**(1/2)*(zeta(z)-1)/((zeta(z)+1)**2 * ((zeta(z) + 1j)*(zeta(z)-1j))**(1/2))

def dz2(z):
    return -1j*hw*(2)**(1/2)*((2*zeta(z)**3 -3*zeta(z)**2 - 3)/((zeta(z) +1)**3 * ((zeta(z) + 1j)*(zeta(z)-1j))**(3/2)))

def dz3(z):
    return 3j*hw*(2)**(1/2)*((2*zeta(z)**5 -4*zeta(z)**4 - zeta(z)**3 - 9*zeta(z)**2 - zeta(z) - 3)/((zeta(z) +1)**4 * ((zeta(z) + 1j)*(zeta(z)-1j))**(5/2)))

def dz4(z):
    return -3j*hw*(2)**(1/2)*((8*zeta(z)**7-20*zeta(z)**6-12*zeta(z)**5 -75*zeta(z)**4 - 20*zeta(z)**3 - 46*zeta(z)**2 - 11)/((zeta(z) +1)**5 * ((zeta(z) + 1j)*(zeta(z)-1j))**(7/2)))

def r1(z):
    return dz1(z)

def r2(z):
    return dz2(z)/(2*dz1(z))

def r3(z):
    return -(dz2(z)**2)/(4*dz1(z)**3) + dz3(z)/(6*dz1(z)**2)

def r4(z):
    return (dz2(z)**3)/(4*dz1(z)**5) - (dz2(z)*dz3(z))/(4*dz1(z)**4) + 1/24 * (dz4(z))/(dz1(z)**3)

def a1(z):
    return 1/dz1(z)

def a2(z):
    return -dz2(z)/(2*dz1(z)**3)

def A(z,alpha):
    return np.conjugate(- lam(alpha)/(dz1(z)))

def B(z,alpha):
    return np.conjugate(lam(alpha) * dz2(z)/(dz1(z)**3))

def Aq(z,alpha):
    return np.conjugate((2*e**2*mu(alpha))/(dz1(z)**3))

def Bq(z,alpha):
    return -3/2 * Aq(z,alpha) * np.conjugate(dz2(z)/dz1(z))

def Cq(z,alpha):
    return -1/2 * Aq(z,alpha) *  np.conjugate(dz3(z)/dz1(z)) - Bq(z,alpha) * np.conjugate(dz2(z)/dz1(z))

def Cs(z,alpha): #z in my scribbles
    return mu(alpha)/dz1(z)

def As(z,alpha): #x in my scribbles
    return np.conjugate((((1+zeta(z))*(1+zeta(z)**2))/(1-zeta(z)))*Cs(z,alpha) + (mu(alpha)*np.conjugate(z))/(dz1(z)**2))

def Bs(z,alpha): #y in my scribbles
    return np.conjugate(-np.conjugate(As(z,alpha)) * dz2(z)/dz1(z) + ((1+zeta(z))*(1+zeta(z)**2))/((1-zeta(z))) * Cs(z,alpha) * dz2(z)/dz1(z) \
                        + ((1+2*zeta(z)+3*zeta(z)**2)/(1-zeta(z))) * Cs(z,alpha))

def f0(z,alpha):
    return A(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**2) + B(z,alpha)/(1/zeta(z)-np.conjugate(zeta(z))) + Aq(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**3) \
    + Bq(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**2) + Cq(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))) \
    + As(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**2) + Bs(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))) \
    + Cs(z,alpha) * r2(z)

def f1(z,alpha):
    return (1/(zeta(z)**2 * dz1(z)))*(2*A(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**3) + B(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**2) \
             + 3*Aq(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**4) + 2*Bq(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**3) \
             + Cq(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**2) + 2*As(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**3) \
             + Bs(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**2)) + Cs(z,alpha) * r3(z)

def g0(z,alpha):
    return ((1+zeta(z))*(1+zeta(z)**2))/(zeta(z)**2 * (1-zeta(z))) * (2*A(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**3) + B(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**2) \
                                                                       + 3*Aq(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**4) + 2*Bq(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**3) + Cq(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**2)\
                                                                       + 2*As(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**3) + Bs(z,alpha)/((1/zeta(z)-np.conjugate(zeta(z)))**2)) \
        + np.conjugate(A(z,alpha)) * (2*r1(z)*r3(z) + r2(z)**2) + np.conjugate(B(z,alpha)) * r2(z) + np.conjugate(Cq(z,alpha)) * r2(z) + np.conjugate(Bq(z,alpha))*(2*r1(z)*r3(z) + r2(z)**2)\
        + np.conjugate(Aq(z,alpha)) * (dz4(z)/(8*dz1(z)) - (dz2(z)*dz3(z))/(4*dz1(z)**2) + dz2(z)**3/(8*dz1(z)**3)) \
        + np.conjugate(Bs(z,alpha))*r2(z) + np.conjugate(As(z,alpha))*(r2(z)**2 + 2*r1(z)*r3(z)) + np.conjugate(Cs(z,alpha))/(1/zeta(z)-np.conjugate(zeta(z))) \
        - ((1+zeta(z))*(1+zeta(z)**2))/(1-zeta(z)) * Cs(z,alpha)*(2*r1(z)*r3(z) + r2(z)**2) - Cs(z,alpha) * (2*r1(z)*r2(z))*a1(z) * (1+2*zeta(z) + 3*zeta(z)**2)/(1-zeta(z)) \
        - Cs(z,alpha) * r1(z)**2 * (a2(z)*(1+2*zeta(z) + 3*zeta(z)**2) + a1(z)**2*(1+3*zeta(z)))/(1-zeta(z))
 
def my_system(x):
    z0_real, z0_imag, alpha = x

    z = z0_real + 1j * z0_imag

    f_val = -f0(z,alpha) + z*np.conjugate(f1(z,alpha)) + np.conjugate(g0(z,alpha))
    dalpha_val = -2*np.imag(f1(z,alpha))

    return [f_val.real, f_val.imag, dalpha_val]

# Initial guess for z0 and alpha
#z0 = 0.7 +0.7 *1j
#z0 = 0.3 + 0.3*1j
#z0 = 0.55 + 0.50 *1j
#a0 = 3*np.pi/4
#z0 =1j*(e*(3/2)**(1/2)) #away from wall it works ie -10 does but close to slit it goes wrong
z0 = 0.1
a0 = np.pi/4
z0 = 0.1 + 1.6*1j
a0 = -3*np.pi/8
z0 = 0 + 1.06*1j
a0 = -np.pi/2
#z0 = 0 + 0.11*1j
#a0 = 0
initial_guess = [z0.real, z0.imag, a0]

# Solve the system using the Newton method  
result = root(my_system, initial_guess, method='hybr')

## Check if the optimization was successful
if result.success:
    print(f"Optimization successful. {result.message}")
    # Extract the solutions
    z0_solution = result.x[0] + 1j * result.x[1]
    alpha_solution = result.x[2]
    print(f"z0 solution: {z0_solution}")
    print(f"alpha solution: {alpha_solution}")
else:
    print(f"Optimization failed. {result.message}")
    z0_solution = result.x[0] + 1j * result.x[1]
    alpha_solution = result.x[2]
    print(f"z0 solution: {z0_solution}")
    print(f"alpha solution: {alpha_solution}")

# Additional information from the result object
print(f"Additional information: {result.message}")