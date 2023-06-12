import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import scipy.integrate
from numpy import fft
import pandas as pd
import time
print(scipy.__version__)


# define parameters and functions:
alpha = .95
epsilon = .1 #.1 gives things that look like solitons
L = np.pi*2
V = -1
D_ = 0.6 #very intereting with D=0.5 (which is the same as Ma=1/4) and D = 0.6

N = 200
Nx = N
n = np.arange(N)
n[int(N/2)+1:] -= N
def RHS_static_cylinder(t,eta):
    eta_z = fft.ifft(2*n*np.pi/L*1j*fft.fft(eta))
    eta_zzz = fft.ifft((n*2*np.pi/L*1j)**3*fft.fft(eta))
    deta_dt = -1/3*fft.ifft(2*n*np.pi/L*1j*fft.fft(eta**3*(1 + D_*eta_z + epsilon**2*eta_zzz))).real
    return deta_dt

# initial condition
Lambda = L
T = 23
k = int(1/np.sqrt(2)/epsilon)
print(k)
z = np.linspace(-L/2, L/2, Nx)
t = np.linspace(0,T, 600)
t_span = (0,T)
u0 = (1 + np.sin(z*k)*0.95)*(1-alpha)*15
print(np.trapz(u0, z))

st = time.time()
print('Starting solve')
result_static = scipy.integrate.solve_ivp(RHS_static_cylinder, t_span, u0, 'BDF', atol = 1e-5, t_eval = t)
u_static = result_static.y
print(np.shape(u_static), ' solved static')
et = time.time()
print(f'Time to solve {et-st} s')

soliton_ic = u_static[:,-1]

# non-deterministic simulation
alpha = .95; epsilon = .1; L = np.pi*2; D = 0.6 #very intereting with D=0.5 (which is the same as Ma=1/4) and D = 0.6
Ma  = (1-D)/2
I_0 = 1; remainder = Ma/I_0**2
N = len(z); n = np.arange(N); n[int(N/2)+1:] -= N

def RHS_not_stoch(time,eta):
    eta_z = fft.ifft(n*1j*np.pi*2/L*fft.fft(eta))
    eta_zzz = fft.ifft((n*1j*np.pi*2/L)**3*fft.fft(eta))
    detadt = -fft.ifft(n*1j*2*np.pi/L*fft.fft(eta**3*(1+(1-2*Ma)*eta_z+epsilon**2*eta_zzz))).real/3
    return detadt

T = 3
t = np.linspace(0,T, 100)
t_eval = np.linspace(0,T, 10)
delta_t = t[1]-t[0]

t_span = (0,T)
u0 = soliton_ic
print(np.trapz(u0, z))
print(L*(1-alpha))
print('Starting solve')
result_ivp_determininistic = scipy.integrate.solve_ivp(RHS_not_stoch, t_span, u0, 'BDF', atol = 1e-5, t_eval =t)
u_det = result_ivp_determininistic.y
print('Solved deterministic problem')
np.save(f'det_soliton_data.npy', u_det)
np.save(f'times_soliton.npy', t_span)
np.save(f'collocation_points_soliton.npy', z)

volatility_list = np.array([0.005, 0.075])
Number_of_iter = 2

big_array = np.zeros((len(volatility_list),np.shape(u_det)[0], np.shape(u_det)[-1], Number_of_iter))
big_currents_array = np.zeros((len(volatility_list),len(t), Number_of_iter))
print(np.shape(big_array))
print(np.shape(big_currents_array))
t_eval = np.linspace(0,T, 10)
print('Starting stochastic iteration')
for k,vol in enumerate(volatility_list):
    results_array = np.zeros((np.shape(u_det)[0], np.shape(u_det)[-1], Number_of_iter))
    currents_array = np.zeros((len(t), Number_of_iter))
    N = len(z); n = np.arange(N); n[int(N/2)+1:] -= N
    st = time.time(); stc = time.process_time()
    print('Started iterating', vol)
    for j in range(Number_of_iter):
        #generate the Brownian Motion for the current 
        I_t = np.cumsum(np.random.normal(loc=0.0, scale = vol*delta_t, size=(len(t), ))) + I_0
        currents_array[:,j] = I_t
        def RHS_no_current(time,eta):
            I = np.interp(time, t,I_t)
            Ma = remainder*I**2
            eta_z = fft.ifft(n*1j*np.pi*2/L*fft.fft(eta))
            eta_zzz = fft.ifft((n*1j*np.pi*2/L)**3*fft.fft(eta))
            detadt = -fft.ifft(n*1j*2*np.pi/L*fft.fft(eta**3*(1+(1-2*Ma)*eta_z+epsilon**2*eta_zzz))).real/3
            return detadt
        result_ivp = scipy.integrate.solve_ivp(RHS_no_current, t_span, u0, 'BDF', atol = 1e-2, t_eval = t)
        results_array[:,:, j] = result_ivp.y

        
        if (j+1) % 10 == 0:
            print('Solved stoch,', np.shape(result_ivp.y), int(100*(j+1)/Number_of_iter), ' %', round(time.time()-st, 3), ' seconds')
        del I_t
        del RHS_no_current
        del result_ivp
    big_array[k, :,:,:] = results_array
    big_currents_array[k,:,:] = currents_array
    etc = time.process_time()

    print(f'{int(100*(k+1)/len(volatility_list))} %')

    # get execution time
    res = etc - stc
    print(f'CPU time = {res}. Per iteration = {round(res/Number_of_iter, 3)}')
    et = time.time()

    # get the execution time
    elapsed_time = et - st

    print(f'Wall time = {elapsed_time}. Per iteration = {round(elapsed_time/Number_of_iter, 3)}')
    
np.save('results.npy', big_array)
np.save('currents.npy', big_currents_array)
np.save('volatilities.npy', volatility_list)