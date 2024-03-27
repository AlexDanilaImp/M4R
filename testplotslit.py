'''main slit plot im using as of 20/02/2024'''
import numpy as np
import matplotlib.pyplot as plt

e = 0.2
h = 0.001 #step size
v1 = 0
v2 = 1
v3 = 0
hw = 100#height of wall

def mu(alpha):
    return e * v2 * np.exp(2j * alpha)

def lam(alpha):
    return -(e**2) * v1  * np.exp(1j * alpha)

def zeta(z):
    return -(z**2)/(z**2+2*hw**2) + (1/4 * ((2*z**2)/(z**2+2*hw**2))**2 - 1)**(1/2)

def dz1(z):
    return 1j*hw*(2)**(1/2)*(zeta(z)-1)/((zeta(z)+1)**2 * (zeta(z)**2 + 1)**(1/2))

def dz2(z):
    return -1j*hw*(2)**(1/2)*((2*zeta(z)**3 -3*zeta(z)**2 - 3)/((zeta(z) +1)**3 * (zeta(z)**2+1)**(3/2)))

def dz3(z):
    return 3j*hw*(2)**(1/2)*((2*zeta(z)**5 -4*zeta(z)**4 - zeta(z)**3 - 9*zeta(z)**2 - zeta(z) - 3)/((zeta(z) +1)**4 * (zeta(z)**2+1)**(5/2)))

def dz4(z):
    return -3j*hw*(2)**(1/2)*((8*zeta(z)**7-20*zeta(z)**6-12*zeta(z)**5 -75*zeta(z)**4 - 20*zeta(z)**3 - 46*zeta(z)**2 - 11)/((zeta(z) +1)**5 * (zeta(z)**2+1)**(7/2)))

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

def A(z,alpha): #dipole
    return np.conjugate(- lam(alpha)/(dz1(z)))

def B(z,alpha): #dipole
    return np.conjugate(lam(alpha) * dz2(z)/(dz1(z)**3))

def Aq(z,alpha):
    return np.conjugate((2*e**2*mu(alpha))/(dz1(z)**3))

def Bq(z,alpha):
    return -3/2 * Aq(z,alpha) * np.conjugate(dz2(z)/dz1(z))

def Cq(z,alpha):
    return -1/2 * Aq(z,alpha) *  np.conjugate(dz3(z)/dz1(z)) - Bq(z,alpha) * np.conjugate(dz2(z)/dz1(z))

def Cs(z,alpha):
    return mu(alpha)/dz1(z)

def As(z,alpha):
    return np.conjugate((((1+zeta(z))*(1+zeta(z)**2))/(1-zeta(z)))*Cs(z,alpha) + (mu(alpha)*np.conjugate(z))/(dz1(z)**2))

def Bs(z,alpha):
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

def euler(z0,a0,h,iterations):
    z_values = [z0]
    a_values = [a0]

    for i in range(iterations):
        if (np.imag(z_values[-1]) <= hw) & (-e <= np.real(z_values[-1]) <= e):
            print("!!!hit the slit!!!")
            break
        elif np.imag(z_values[-1]) <= e:
            print("!!!hit bottom wall!!!")
            break
        elif (hw<=np.imag(z_values[-1])<= hw+e) & ((np.real(z_values[-1])**2 + (np.imag(z_values[-1])-hw)**2)**(1/2) <= e):
            print("!!!hit the slit from above!!!")
            break
        else:
            z_next = z_values[-1] + h * dzdt(z_values[-1],a_values[-1])
            a_next = a_values[-1] + h * dadt(z_values[-1],a_values[-1])
            z_values.append(z_next)
            a_values.append(a_next)
    
    return z_values,a_values

num_iterations = 10000
#a = -np.pi/8
#z1 = -2 + 5j*e
#z1 =1j*(e*(3/2)**(1/2)) #away from wall it works ie -10 does but close to slit it goes wrong
#a = -np.pi/4
#z1 = 0.7 + 0.35j*2
#z1 = 2**(1/2)/2 * (1+1j)
#z1 = 0.1 + 1.6*1j
#a = -3*np.pi/8 #- tell Sam about Davis & Crowdy 2012
#z1 = -1 + 1j*2
#z1 =  100 + 1j*(e*(3/2)**(1/2)) + 1j
#a = -np.pi/4
z1 = 2**(1/2)/2 * (1+1j)
a = np.pi/2
zval1, aval1 = euler(z1,a ,h,num_iterations)
xval1 = [np.real(z) for z in zval1]
yval1 = [np.imag(z) for z in zval1]
xarray1 = np.array(xval1)
yarray1 = np.array(yval1)
array1 = np.array(aval1)
arrow_position = 1  # Change this index to position the arrow

# Plotting the vertical line
plt.axvline(x=0, ymin=0, ymax=2, color='red', linestyle='--', label='Vertical Line')

# Plotting the horizontal line
plt.axhline(y=e,xmin=min(xarray1),xmax= max(xarray1), color='red', linestyle='--', label='Horizontal Line')

# Plotting the horizontal line
plt.axhline(y=e,xmin=min(xarray1),xmax= max(xarray1), color='red', linestyle='--', label='Horizontal Line')
#y = e+hw

# Plot the line
plt.plot(xarray1, yarray1, color='blue')

# Add arrowhead at a specified position
plt.annotate(
    '', xy=(xarray1[arrow_position], yarray1[arrow_position]), xytext=(xarray1[arrow_position - 1], yarray1[arrow_position - 1]),
    arrowprops=dict(facecolor='red', arrowstyle='->, head_width = 0.5, head_length = 0.5', color = "blue")
)

plt.xlabel("x")
plt.ylabel('y')
plt.show()

print(min(xarray1))
print(max(xarray1))