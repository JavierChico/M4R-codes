import numpy as np
import scipy.linalg
import scipy.integrate
from scipy import fft
import matplotlib.pyplot as plt



# define parameters and functions:
alpha = .5
epsilon = .1
L = 2*np.pi
tol = 0.0001

D_list = np.linspace(0, .1, 10)


T = 2
k = int(1/np.sqrt(2)/epsilon)
print(k)
z = np.linspace(0, 2*np.pi, 300)
t = np.linspace(0,T, 50)
t_span = (0,T)
u0 = (1 + np.sin(z*k)*0.95)*(1-alpha)+alpha

results = np.zeros((len(z), len(t), len(D_list)))
N = len(z)
n = np.arange(N)
n[int(N/2)+1:] -= N
for j, D in enumerate(D_list):
    def RHS(t,S):
        Ma = (1-D)/2
        p = 1/S - epsilon**2*fft.ifft(n**2*(2*np.pi/L*1j)**2*fft.fft(S)) -1/S**2*Ma 
        p_z = fft.ifft(n*(2*np.pi/L*1j)*fft.fft(p))
        factor = 2*S**2*(alpha**2 - S**2 + 2*S**2*np.log(S/alpha)) - (alpha**2 - S**2)**2
        dSdt = fft.ifft(n*2*np.pi/L*1j*fft.fft((p_z - 1)*factor)).real/3/S/16 #(S^2)_t

        return dSdt
    result_ivp = scipy.integrate.solve_ivp(RHS, t_span, u0, 'BDF', atol = 1e-7, t_eval =t)
    u = result_ivp.y
    results[:,:,j] = u
    print('Done here')

Z,T = np.meshgrid(z,t)
plt.figure()
plt.contourf(Z,T,u.T)
plt.colorbar()
plt.show()