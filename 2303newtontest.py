import numpy as np

e = 0.2
h = 0.001 #step size
v1 = 0
v2 = 1
v3 = 0
z0 = 5j*e
hw = 1#height of wall

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
        + np.conjugate(Aq(z,alpha)) * (3*r1(z)**2 * r4(z) + 6*r1(z)*r2(z)*r3(z) + r2(z)**3) \
        + np.conjugate(Bs(z,alpha))*r2(z) + np.conjugate(As(z,alpha))*(r2(z)**2 + 2*r1(z)*r3(z)) + np.conjugate(Cs(z,alpha))/(1/zeta(z)-np.conjugate(zeta(z))) \
        - ((1+zeta(z))*(1+zeta(z)**2))/(1-zeta(z)) * Cs(z,alpha)*(2*r1(z)*r3(z) + r2(z)**2) - Cs(z,alpha) * (2*r1(z)*r2(z))*a1(z) * (1+2*zeta(z) + 3*zeta(z)**2)/(1-zeta(z)) \
        - Cs(z,alpha) * r1(z)**2 * (a2(z)*(1+2*zeta(z) + 3*zeta(z)**2) + a1(z)**2*(1+3*zeta(z)))/(1-zeta(z))
 
def dzdt(z,alpha):
    return -f0(z,alpha) + z*np.conjugate(f1(z,alpha)) + np.conjugate(g0(z,alpha))   

def dadt(z,alpha):
    return -2*np.imag(f1(z,alpha))

def dxdtdx(z,alpha,step):
    return np.real((dzdt(z+step,alpha) - dzdt(z-step,alpha))/(2*step))

def dxdtdy(z,alpha,step):
    return np.real((dzdt(z+step*1j,alpha) - dzdt(z-step*1j,alpha))/(2*step))

def dxdtda(z,alpha,step):
    return np.real((dzdt(z,alpha+step*0.1) - dzdt(z,alpha-step))/(2*step*0.1))

def dydtdx(z,alpha,step):
    return np.imag((dzdt(z+step,alpha) - dzdt(z-step,alpha))/(2*step))

def dydtdy(z,alpha,step):
    return np.imag((dzdt(z+step*1j,alpha) - dzdt(z-step*1j,alpha))/(2*step))

def dydtda(z,alpha,step):
    return np.imag((dzdt(z,alpha+step*0.1) - dzdt(z,alpha-step))/(2*step*0.1))

def dadtdx(z,alpha,step):
    return (dadt(z+step,alpha)-dadt(z-step,alpha))/(2*step)

def dadtdy(z,alpha,step):
    return (dadt(z+step*1j,alpha)-dadt(z-step*1j,alpha))/(2*step)

def dadtda(z,alpha,step):
    return (dadt(z,alpha+step*0.1)-dadt(z,alpha-step))/(2*step*0.1)

def Jacobian(z,alpha,step):
    return np.array([[dxdtdx(z,alpha,step), dydtdx(z,alpha,step),dadtdx(z,alpha,step)], [dxdtdy(z,alpha,step), dydtdy(z,alpha,step), dadtdy(z,alpha,step)], [dxdtda(z,alpha,step), dydtda(z,alpha,step), dadtda(z,alpha,step)]])

error = 100
tol = 0.00000732
maxiter = 100
step = 0.01
i=0
z0 = 1 + 1j
a0 = np.pi/4
z0 = 0 + 0.11*1j
a0 = 0
z0 = 0+ 1.06*1j
a0 = -np.pi/2
#z0 = 0.01 + 0.01j
#a0 = -np.pi/4
x0 = np.array([np.real(z0),np.imag(z0),a0]).reshape(3,1)


while np.any(abs(error) > tol) and i<maxiter:

    fun_evaluate = np.array([np.real(dzdt(x0[0]+ x0[1]*1j,x0[2])), np.imag(dzdt(x0[0]+ x0[1]*1j,x0[2])), dadt(x0[0]+ x0[1]*1j,x0[2])])
    print(i)
    print(abs(error))
    print("--------")
    flat_x0 = x0.flatten()
    x_new = x0 - np.linalg.inv(Jacobian(flat_x0[0]+ flat_x0[1]*1j,flat_x0[2],step))@fun_evaluate

    error = x_new - x0

    x0 = x_new

    i = i+1

print("The solution is")
print(x_new)
print("Eq-1", np.around(np.real(dzdt(x_new[0]+ x_new[1]*1j,x_new[2])),3))
print("Eq-2", np.around(np.imag(dzdt(x_new[0]+ x_new[1]*1j,x_new[2])),3))
print("Eq-3", np.around(dadt(x_new[0]+ x_new[1]*1j,x_new[2]),3))