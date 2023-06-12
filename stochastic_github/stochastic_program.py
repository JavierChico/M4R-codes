import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.integrate
from matplotlib import cm
from scipy import fft
#from matplotlib.animation import FuncAnimation, PillowWriter 

alpha = .95; epsilon = .3; L = 2*np.pi; Ma = 0.5; vol = 0.1; I_0 = 1; remainder = Ma/I_0**2

def RHS_not_stoch(time,eta):
    N = len(eta)
    n = np.arange(N);
    n[int(N/2)+1:] -= N
    eta_z = fft.ifft(n*1j*np.pi*2/L*fft.fft(eta))
    eta_zzz = fft.ifft((n*1j*np.pi*2/L)**3*fft.fft(eta))
    detadt = -fft.ifft(n*1j*2*np.pi/L*fft.fft(eta**3*(1+(1-2*Ma)*eta_z+epsilon**2*eta_zzz))).real/3
    return detadt

T = 30
k = 1/np.sqrt(2)/epsilon
print(k)
z = np.linspace(0, 2*np.pi, 200)
t = np.linspace(0,T, 300)
t_eval = np.linspace(0,T, 10)
delta_t = t[1]-t[0]

t_span = (0,T)
u0 = (1.1 - np.cos(z*k))/2 
result_ivp_determininistic = scipy.integrate.solve_ivp(RHS_not_stoch, t_span, u0, 'BDF', atol = 1e-9, t_eval =t)

u_det = result_ivp_determininistic.y
print('Solved deterministic ,', np.shape(u_det))
max_value_deterministic = np.max(u_det, axis=0)[-1]

Number_of_iter = 100

max_values = np.zeros(Number_of_iter)
mean_current = np.zeros(Number_of_iter)

for j in range(Number_of_iter):
    #generate the Brownian Motion for the current 
    I_t = np.cumsum(np.random.normal(loc=0.0, scale = vol*delta_t, size=(len(t), ))) + I_0
    
    mean_current[j] = np.mean(I_t)
    def RHS_no_current(time,eta):
        I = np.interp(time, t,I_t)
        Ma = remainder*I**2
        N = len(eta)
        n = np.arange(N);
        n[int(N/2)+1:] -= N
        eta_z = fft.ifft(n*1j*np.pi*2/L*fft.fft(eta))
        eta_zzz = fft.ifft((n*1j*np.pi*2/L)**3*fft.fft(eta))
        detadt = -fft.ifft(n*1j*2*np.pi/L*fft.fft(eta**3*(1+(1-2*Ma)*eta_z+epsilon**2*eta_zzz))).real/3
        return detadt

    result_ivp = scipy.integrate.solve_ivp(RHS_no_current, t_span, u0, 'BDF', atol = 1e-7, t_eval = t_eval)
    max_values[j] = np.max(result_ivp.y, axis = 0)[-1]
    
    if (j+1) % 10 == 0:
        print('Solved stoch,', np.shape(u), int(100*(j+1)/Number_of_iter), ' %')

plt.plot(mean_current, max_values, '.')
plt.scatter(1, max_value_deterministic)
plt.grid()
plt.title(f'Maximum value in the profile vs mean current in the Brownian Motion Realization, vol = {vol}')
plt.xlabel('Mean Current')
plt.ylabel('Maximum value at the end of the time evolution')
plt.savefig('stoch_max_vs_mean_current.jpg')