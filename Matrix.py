import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import time 

start_time = time.time()

# Parameters
beta = 0.6
m = 1.0   # mass
k = 1.0   # spring constant
Lambda = 5  # path length
a = beta / Lambda # Lattice Spacing
n =  15# matrix size
iterations = 3000  # number of HMC iterations to run
burn_in = int(iterations / 10)
leap_step = 0.07
g = np.identity(n)

g = np.identity(n)
g_inv=inv(g)



def unblock(x):
    n = len(x)
    L = int(len(x[0]) / n)
    X = [x[:, i*n:(i+1)*n] for i in range(L)]
    return X

def F(x, a):  # force corresponding to displacement in SHM
    V = Action_Matrix(x, a)
    return (-1 / a) * (x @ V)


def Leap(x, p, space, N=5):  # Leapfrog integration
    for i in range(N):
        phalf = p + space / 2 * F(x, space)
        x = x + space * phalf 
        p = phalf + space / 2 * F(x, space)
        for j in range(Lambda):
            x[:, j*n:(j+1)*n] = (x[:, j*n:(j+1)*n] + x[:, j*n:(j+1)*n].T) / 2
    return x, p

def norm_sym(n,scal):  # matrix generator
    A0 = np.random.normal(loc=0.0, scale=scal, size=(n, n))  # random matrix
    Au = np.triu(A0)  # takes upper triangle 
    Al = np.tril(Au.T, k=-1)  # creates symmetric lower triangle
    A = Au + Al  # sums together
    return A

def mat_array(Lambda, n,scal):  # generates arrays of matrices
    x = []
    for i in range(Lambda):
        x.append(norm_sym(n,scal))  
    return np.block(x)

def Action_Matrix(x, a):
    N = n
    M = np.zeros((Lambda * N, Lambda * N))
    for i in range(Lambda):
        M[i*N:(i+1)*N, i*N:(i+1)*N] = (2 + (a)**2) * np.eye(N)
        if i != Lambda - 1:
            M[i*N:(i+1)*N, (i+1)*N:(i+2)*N] = -np.eye(N)
            M[(i+1)*N:(i+2)*N, i*N:(i+1)*N] = -np.eye(N)
    M[0:N, (Lambda-1)*N:Lambda*N] += -np.eye(N)
    M[(Lambda-1)*N:Lambda*N, 0:N] += -np.eye(N)
    return M

def Action(x, a):  # Piecewise action
    V1 = Action_Matrix(x, a)
    action = (1 / (2 * a)) * np.trace((x @ V1 @ x.T))
    return action

def Ham(x, p, a):
    kinetic = 0.5 * np.trace(p @ p.T)
    return kinetic + Action(x, a)

def Metropolis(x, space):
    acceptance = 0
    rand_p = mat_array(Lambda, n,1)
    x_prop, p_prop = Leap(x, rand_p, space)
    Ham_current = Ham(x, rand_p, space)
    Ham_prop = Ham(x_prop, p_prop, space)
    alpha = np.exp(Ham_current - Ham_prop)
    if np.random.uniform(0, 1) <= alpha:
        acceptance = 1
        return x_prop, acceptance	
    return x, acceptance

def wigner(R, x):
    return (2 / (np.pi * R**2)) * np.sqrt(R**2 - x**2)

# Initialize configuration
x = mat_array(Lambda, n,3)


n = len(x)  # size of square matrices comprising x
L = int(len(x[0]) / n)  # number of matrices in x array
hamiltons = 0
ham_matrix = []
no_accepted = 0
S = [Action(x, a)]
eigen = [np.linalg.eigvals(unblock(x))]
ham_actual = ((n**2)/2 + m/(np.exp(-beta) -1))#m * (1/2 + (1)/(np.exp(m * beta) -1))

for i in range(1, iterations):
    x, acceptance = Metropolis(x, leap_step)
    no_accepted += acceptance
    S.append(Action(x, a))
    path = unblock(x)
    eigen.append(np.linalg.eigvals(path))
    hamiltons = 0
    '''for j in range(Lambda):
        hamiltons += np.trace(path[j] @ path[j])
    ham_matrix.append(1/( Lambda) * hamiltons )'''




acc_rate = no_accepted / iterations * 100
S, iterations = S[burn_in:-1], iterations-burn_in

# Plotting the action vs. Monte Carlo iterations
plt.figure(figsize=(12, 6))
plt.plot(S, label='HMC')
plt.xlabel('Monte Carlo iterations')
plt.ylabel('<S>')
plt.axhline(y = (n**2 *Lambda)/2, color= 'orange')
plt.grid(True)
plt.legend()
plt.show()

'''plt.plot(ham_matrix)
plt.xlabel('Monte Carlo iterations')
plt.ylabel('Hamiltonian <H>')
plt.axhline(y=ham_actual, color='orange')
plt.grid(True)
plt.show()'''

eigen = np.ndarray.flatten(np.array(eigen))
plt.figure()
plt.hist(eigen, 30, density=True)
plt.xlabel('Eigenvalues')
plt.ylabel('Density')
plt.title('Histogram of Eigenvalues')
plt.show()
# Plotting the Wigner semicircle distribution
'''plt.plot(x_wig, wig, color='orange')
plt.xlabel('$x$')
plt.ylabel('$W(x)$')
plt.title('Wigner Semicircle Distribution')'''


print('Acceptance rate: ', acc_rate, '%')
print('Expectation value of <H>: ', np.sum(ham_matrix) / iterations)
print('Expectation value should be: ', ham_actual)

print('Expectation value of <S>: ', np.sum(S) / iterations)
print('Expectation value should be: ', (n**2 *Lambda)/2)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

