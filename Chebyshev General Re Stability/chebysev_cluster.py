import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as cheb
import scipy
#import scipy as sp
#from scipy.optimize import fsolve
import scipy.linalg
print(scipy.__version__)

alpha = 0.9; Bo=0.1; Ca=0.1; Re=0.01; Ma=0; epsilon=0.1
def w_0(x):
    r = ((1-alpha)*x+1+alpha)/2
    return 1/4*(2*np.log(r/alpha)-(r**2-alpha**2))
def w_0_prime(x):
    r = ((1-alpha)*x+1+alpha)/2
    return 1/2*(1/r-r)
w_0 = np.vectorize(w_0)
w_0_prime = np.vectorize(w_0_prime)

N = 128
x = np.flip(cheb.chebgauss(N+1)[0])
x_interior = x[1:-1]
matrix = np.zeros((N+1, N+1))
#build the polys at the points
cheb_poly = np.zeros((N+1, N+1))
cheb_der = np.zeros_like(cheb_poly)
cheb_der_der = np.zeros_like(cheb_poly)
# row is the order, column is the evaluation point
cheb_poly[0,:] = 1
cheb_poly[1,:] = x
cheb_der[1,:] = 1
for k in range(2,N+1):
    cheb_poly[k,:] = 2*x*cheb_poly[k-1,:] - cheb_poly[k-2,:]
    cheb_der[k,:] = 2*cheb_poly[k-1] + 2*x*cheb_der[k-1, :] - cheb_der[k-2, :]
    cheb_der_der[k,:] = 4*cheb_der[k-1]+2*x*cheb_der_der[k-1, :]-cheb_der_der[k-2, :]
print('Polynomials done')
kappa_vect = np.linspace(0, 1, 20)
Lambda_vect = np.zeros_like(kappa_vect)
r_interior = (1-alpha)*x_interior + (1+alpha) #(half)

for index, kappa in enumerate(kappa_vect):
    #continuity first
    matrix_a_cont = np.zeros((N-1, N+1), dtype=complex)
    matrix_b_cont = np.zeros((N-1, N+1), dtype=complex)
    matrix_c_cont = np.zeros((N-1, N+1), dtype=complex)
    matrix_S_cont = np.zeros((N-1, 1), dtype = complex)
    
    matrix_a_cont += np.transpose(2/(1-alpha)*cheb_der[:,1:-1] + 2/(r_interior)*cheb_poly[:,1:-1])
    matrix_b_cont += np.transpose(1j*kappa*cheb_poly[:, 1:-1])

    matrix_cont = np.concatenate([matrix_a_cont, matrix_b_cont, matrix_c_cont, matrix_S_cont], axis=1)
    
    matrix_cont_forcing = np.zeros_like(matrix_cont)
    # Ns in r direction (or x direction now)
    matrix_a_NSr = np.zeros((N-1, N+1), dtype=complex)
    matrix_b_NSr = np.zeros((N-1, N+1), dtype=complex)
    matrix_c_NSr = np.zeros((N-1, N+1), dtype=complex)
    matrix_S_NSr = np.zeros((N-1, 1), dtype = complex)
    matrix_a_NSr_forcing = np.zeros((N-1, N+1), dtype=complex)
    
    matrix_a_NSr += np.transpose(Re*w_0(x_interior)*1j*kappa*cheb_poly[:, 1:-1] + kappa**2*cheb_poly[:, 1:-1])
    matrix_a_NSr += np.transpose(-4/(1-alpha)**2*cheb_der_der[:,1:-1] + 4/(r_interior**2)*cheb_poly[:,1:-1])
    matrix_a_NSr += np.transpose(-2*cheb_der[:,1:-1]/r_interior)
    
    matrix_c_NSr += np.transpose(2/(1-alpha)*cheb_der[:,1:-1])
    
    matrix_a_NSr_forcing += np.transpose(-Re*cheb_poly[:,1:-1])
    
    matrix_NSr = np.concatenate([matrix_a_NSr, matrix_b_NSr, matrix_c_NSr, matrix_S_NSr], axis=1)

    matrix_NSr_forcing = np.concatenate([matrix_a_NSr_forcing, np.zeros((N-1, N+1), dtype=complex), 
                                         np.zeros((N-1, N+1), dtype=complex), 
                                         np.zeros((N-1, 1), dtype = complex)], axis=1)
    
    #NS in z
    matrix_a_NSz = np.zeros((N-1, N+1), dtype=complex)
    matrix_b_NSz = np.zeros((N-1, N+1), dtype=complex)
    matrix_c_NSz = np.zeros((N-1, N+1), dtype=complex)
    matrix_S_NSz = np.zeros((N-1, 1), dtype = complex)
    matrix_b_NSz_forcing = np.zeros((N-1, N+1), dtype=complex)
    
    matrix_a_NSz += np.transpose(Re*w_0_prime(x_interior)*cheb_poly[:,1:-1])
    matrix_b_NSz += np.transpose(Re*1j*kappa*w_0(x_interior)*cheb_poly[:,1:-1]+kappa**2*cheb_poly[:,1:-1])
    matrix_b_NSz += np.transpose(-4/(1-alpha)**2*cheb_der_der[:,1:-1]-2/r_interior*cheb_der[:,1:-1])
    matrix_c_NSz += np.transpose(1j*kappa*cheb_poly[:,1:-1])
    
    matrix_b_NSz_forcing += np.transpose(-Re*cheb_poly[:,1:-1])
    matrix_NSz = np.concatenate([matrix_a_NSz, matrix_b_NSz, matrix_c_NSz, matrix_S_NSz], axis=1)
    
    matrix_NSz_forcing = np.concatenate([np.zeros((N-1, N+1), dtype=complex), matrix_b_NSz_forcing,
                                         np.zeros((N-1, N+1), dtype=complex), 
                                         np.zeros((N-1, 1), dtype = complex)], axis=1)
    # u(r = alpha) = u(x=-1)=0
    no_slip = np.transpose(cheb_poly[:,0]).reshape(1,-1)
    matrix_noslip_u = np.concatenate([no_slip,np.zeros_like(no_slip),np.zeros_like(no_slip)], axis=1)
    # no slip w
    matrix_noslip_w = np.concatenate([np.zeros_like(no_slip),no_slip,
                                      np.zeros_like(no_slip)], axis=1)
    # du/dr=0 at r=alpha (du/dx=0 at x=-1)
    derivative = np.transpose(cheb_der[:,0]).reshape(1, -1)
    matrix_no_derivative_u = np.concatenate([derivative, np.zeros_like(derivative),
                                      np.zeros_like(derivative)], axis=1)
    # join them and add column for S
    BC_at_wire_almost = np.concatenate([matrix_noslip_u, matrix_noslip_w, matrix_no_derivative_u], axis=0)
    matrix_BC_at_wire = np.concatenate([BC_at_wire_almost, np.zeros((3,1))], axis=1)
    matrix_BC_at_wire_forcing = np.zeros_like(matrix_BC_at_wire)

    
    #kinematic 
    length_row = len(np.transpose(cheb_poly[:,-1]).reshape(1,-1))
    condition_for_u = np.transpose(cheb_poly[:,-1]).reshape(1,-1)
    matrix_kinematic=np.concatenate([condition_for_u, 
                                   np.zeros_like(condition_for_u), 
                                   np.zeros_like(condition_for_u), 
                                   np.array([[-w_0(x[-1])*1j*kappa]])], axis=1)
    matrix_kinematic_forcing = np.concatenate([np.zeros_like(condition_for_u), np.zeros_like(condition_for_u),
                                               np.zeros_like(condition_for_u), np.array([[1]])], axis = 1)
    
    # tangential 
    matrix_tangential = np.concatenate([1j*kappa*cheb_poly[:,-1].reshape(1,-1),
                                        2/(1-alpha)*cheb_der[:,-1].reshape(1,-1),
                                        np.zeros_like(condition_for_u),
                                       np.array([[-1]])], axis =1)
    
    #normal
    matrix_normal = np.concatenate([(-4*Ca/(1-alpha))*cheb_der[:,-1].reshape(1,-1), 
                                   np.zeros_like(condition_for_u), 
                                   Bo*cheb_poly[:,-1].reshape(1,-1), np.array([[(1-2*Ma)-kappa**2]])], axis=1)
    matrix_BC_free_surface = np.concatenate([matrix_kinematic, matrix_tangential, matrix_normal], axis=0)
    matrix_BC_free_surface_forcing = np.concatenate([matrix_kinematic_forcing,
                                                     np.zeros_like(matrix_kinematic_forcing),
                                                    np.zeros_like(matrix_kinematic_forcing)], 
                                                    axis=0)
    #continuity at r=1
    
    cont_at_free_surface = np.concatenate([(2/(1-alpha)*cheb_der[:,0] + 2/(x[-1])*cheb_poly[:,0]).reshape(1,-1), 
                                          (1j*kappa*cheb_poly[:, 0]).reshape(1,-1), 
                                          np.zeros_like((1j*kappa*cheb_poly[:, 0]).reshape(1,-1)), 
                                          np.array([[0]])], axis =1)


    
    matrix = np.concatenate([matrix_cont, matrix_NSr, matrix_NSz, 
                             matrix_BC_at_wire, matrix_BC_free_surface, cont_at_free_surface],
                           axis=0)
    matrix_forcing = np.concatenate([matrix_cont_forcing, matrix_NSr_forcing, matrix_NSz_forcing, 
                                    matrix_BC_at_wire_forcing, matrix_BC_free_surface_forcing,
                                     np.zeros_like( cont_at_free_surface)], axis=0)
    #print(np.shape(matrix))
    #print(np.shape(matrix_forcing))
    evals = scipy.linalg.eigvals(matrix,matrix_forcing).real
    print(evals)
    Lambda_vect[index] = np.max(evals[~np.isinf(evals)])
    print((index+1)/len(kappa_vect)*100)
print('Done')
#np.save(f'Lambda_vect_N_{N}.npy', Lambda_vect)