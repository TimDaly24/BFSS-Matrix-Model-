import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
start_time = datetime.now()


iterations = 10000              # number of metropolis iteratoins
burn_in = int(iterations/10)    # how many of the initial iterations will be discarded
Dim = 2                         # number of spacial dimensions
Lambda = 5                     # number of sites on lattice
N = 7                          # size of matrices
beta = 0.3                      #
a = beta/Lambda                 #
m = 1.0                         #
eps =0.00007                       # leapfrog integration step size


def comp_herm_norm(N, standard_deviation):
    """Generates hermitian matrices with complex values from a normal distrubution """
    diag_array = np.random.normal(loc=0.0, scale=standard_deviation, size=N)
    diagonal = np.diag(diag_array)
    
    real_random_matrix = np.random.normal(0.0, standard_deviation/2, (N, N))
    real_random_matrix -= np.diag(real_random_matrix)
    comp_random_matrix = 1j*np.random.normal(0.0, standard_deviation/2, (N, N))
    comp_random_matrix -= np.diag(comp_random_matrix)
    
    upper_triangle = np.triu(real_random_matrix + comp_random_matrix, k=1)
    lower_triangle = upper_triangle.conj().T
    
    return diagonal + upper_triangle + lower_triangle


def position_array(Dim, Lambda, N, standard_deviation=1):
    """Generates path of complex hermitian matrices in 'Dim' dimensions"""
    X = []
    for i in range(Dim):
        x = []
        for i in range(Lambda):
            x.append(comp_herm_norm(N, standard_deviation))
        X.append(x)
    return np.array(X)


def S_b(x, theta):
    """The bosonic action"""
    D = np.diag(np.exp(1j*theta))   # create D matrix from theta
    D_adj = np.conj(D).T            # create adjoint of D
    Dim, Lambda, N = len(x), len(x[0]), len(x[0][0])    # extract shape of x

    A = np.full((N, N), 0+0j)    # initialise array
    
    for n in range(Lambda-1):       # sum over each site
        for i in range(Dim):        # sum over each dimension
            A -=x[i][n]@x[i][n+1]
            
    for i in range (Dim):               # sum over each dimension
        A -= x[i][-1] @ D @ x[i][0] @ D_adj

        for n in range(Lambda):         # sum over each site
            for i in range(Dim):        # sum over each dimension
                A += x[i][n]@x[i][n]   # x^2 term
                for j in range(Dim):                # now commutator term from BFSS
                    #if i!=j:    # commutator cancels out if i=j
                        A -= a*a/4 * np.trace(x[i][n] @ x[j][n] @ x[i][n] @ x[j][n]
                                      +x[j][n] @ x[i][n] @ x[j][n] @ x[i][n]
                                      -x[i][n] @ x[j][n] @ x[j][n] @ x[i][n]
                                      -x[j][n] @ x[i][n] @ x[i][n] @ x[j][n])
                
    return np.real(N*(np.trace(A)/a))


def F_x(x, theta):
    "Force corresponding to position, -∂S_b/∂x_n for n = 0,...,Lambda-1"
    D = np.diag(np.exp(1j*theta))      # create D matrix from theta
    D_adj = np.conj(D).T               # create adjoint of D
    Dim, Lambda = len(x), len(x[0])    # extract shape of x
    
    F = np.full(np.shape(x), fill_value=(0+0j))
    for i in range(Dim):    # for each dimension i = 0,...,Dim
        
        # first term -∂S/∂x_0
        F[i][0] = 1/a * (x[i][1] + D_adj @ x[i][-1] @ D - x[i][0])
        for j in range(Dim):    # commutator term
            if i!=j:            # commutator cancels out if i=j
                F[i][0] += a*(x[j][0] @ x[i][0] @ x[j][0] - x[j][0] @ x[j][0] @ x[i][0]
                             -x[i][0] @ x[j][0] @ x[j][0] + x[j][0] @ x[i][0] @ x[j][0])
        
        # -∂S/∂x_n for n = 1,...,Lambda-2
        for n in range(1, Lambda-1):
            F[i][n] = 1/a * (x[i][n-1] + x[i][n+1] - x[i][n])
            for j in range(Dim):    # commutator term
                if i!=j:            # commutator cancels out if i=j
                        F[i][n] += a*(x[j][n] @ x[i][n] @ x[j][n] - x[j][n] @ x[j][n] @ x[i][n]
                                     -x[i][n] @ x[j][n] @ x[j][n] + x[j][n] @ x[i][n] @ x[j][n])
        
        # last term -∂S/∂x_Lambda-1
        F[i][-1] = 1/a * (x[i][-2] + D@x[i][0]@D_adj - x[i][-1])
        for j in range(Dim):    # commutator term
            if i!=j:            # commutator cancels out if i=j
                F[i][-1] += a*(x[j][-1] @ x[i][-1] @ x[j][-1] - x[j][-1] @ x[j][-1] @ x[i][-1]
                              -x[i][-1] @ x[j][-1] @ x[j][-1] + x[j][-1] @ x[i][-1] @ x[j][-1])
    return N*F


def F_theta(x, theta):
    Dim, N = len(x), len(theta)     # extract shape of x, theta
    
    F = np.zeros(N)
    for l in range(N):          # ranges over entries in F
        for k in range(N):          # sums over k (k replaces m from BFSS paper)
            for i in range(Dim):    # sums over dimensions
                F[l] += (2*N/a)*np.real(1j * x[i][-1][k][l] * x[i][0][l][k] 
                                        *np.exp(1j*( theta[l]-theta[k] )) )    
        
            if k != l:
                F[l] +=1/np.tan((theta[l]-theta[k])/2)
    return F


def S_FP(theta):
    SFP = 0
    N = len(theta)
    for l in range(N):
        for k in range(N):
            if l !=  k:
                SFP -= np.log(abs(np.sin( (theta[l]-theta[k])/2)))
    return SFP


def Hamiltonian(x, p_x, theta, p_theta):
    Dim, Lambda, N = len(x), len(x[0]), len(theta)  # extract shape of x, theta 
    H = 0
    
    for n in range(Lambda):                         # sums over lattice sites
        for i in range(Dim):                        # sums over dimensions
            H += np.trace(p_x[i][n]@p_x[i][n])
        
    for l in range(N):                              # sums over angles in theta
        H += p_theta[l]*p_theta[l]
    
    return H/2 + S_b(x, theta) + S_FP(theta)


def Leap(X, P_X, THETA, P_THETA, space=eps, r=5): # Leapfrog integration
    x = X.copy()
    p_x = P_X.copy()
    phalf_x = x.copy()
    theta = THETA
    p_theta = P_THETA.copy()
    phalf_theta = theta.copy()
# making copies as spyder can erroneously redefine variables globally in a function 
    Dim = len(x)
    
    for q in range(r):          # runs leapfrog r times
        for i in range(Dim):    # ranging over dimensions
            phalf_x[i] = p_x[i] + space/2*F_x(x, theta)[i] 
        phalf_theta = p_theta + space/2*F_theta(x, theta)
        
        for i in range(Dim):    # ranging over dimensions
            x[i] += space*phalf_x[i]
        theta += space*phalf_theta
        
        for i in range(Dim):    # ranging over dimensions
            p_x[i] = phalf_x[i] + space/2*F_x(x, theta)[i]
        p_theta = phalf_theta + space/2*F_theta(x, theta)
    
    return x, p_x, theta, p_theta


def Metropolis(x, theta):
    """Metropolis-Hastings Algorithm"""
    acceptance = 0                                  # will become 1 if state is accepted
    
    rand_p_x = position_array(Dim, Lambda, N, 1)    # random x momentum
    rand_p_theta = np.random.normal(0.0, 1, N)      # random theta momentum
# now it proposes a new state by evloving the state with origanl x, theta and random momenta
    x_prop, p_x_prop, theta_prop, p_theta_prop = Leap(x, rand_p_x, theta, rand_p_theta)
    
    Ham_current = np.real(Hamiltonian(x, rand_p_x, theta, rand_p_theta))
    Ham_prop = np.real(Hamiltonian(x_prop, p_x_prop, theta_prop, p_theta_prop))
    alpha = np.exp(Ham_current - Ham_prop)
    # alpha is probability of accepting new configuration if alpha<1
    
    if np.random.uniform(0,1) <= alpha:
        acceptance = 1
        return x_prop, theta_prop, acceptance# , np.real(np.sum(np.trace(x_prop@x_prop)))
    # decides whether or not to update the configuartion and tracks acceptance
    
    return x, theta, acceptance# , np.real(np.sum(np.trace(x@x))/Lambda)


def eig(x):
    """Sums the absolute values of the eigenvalues of all position matrices a state"""
    return sum(np.ndarray.flatten(abs(np.linalg.eigvals(x))))


def internal_energy(x):
    """Gives the internal energy E/(N^2)"""
    Dim, Lambda, N = len(x), len(x[0]), len(x[0][0])
    A = np.zeros((N, N))            # will use to get commutator term
    
    for n in range(Lambda):         # summing over lattice sites
        for i in range(Dim):        # summing over dimensions
            for j in range(Dim):
                if i!=j:            # commutator cancels out if i=j
                        A += a*(x[j][n] @ x[i][n] @ x[j][n] - x[j][n] @ x[j][n] @ x[i][n]
                                     -x[i][n] @ x[j][n] @ x[j][n] + x[j][n] @ x[i][n] @ x[j][n])
    return -3*a/(4*N*beta)*np.trace(A)


x = position_array(Dim, Lambda, N, 0.04)      # initial position
theta = np.random.normal(0.0, 0.04, N)   # initial theta
no_accepted=0
S = [S_b(x, theta)]
eigen = [np.linalg.eigvals(x)]


for i in range(1, iterations):
    x, theta, acceptance = Metropolis(x, theta)
    no_accepted += acceptance   # track acceptance rate
    S.append(S_b(x, theta))     # keep list of bosonic action at each iteration
    eigen.append(np.linalg.eigvals(x))  # keep traack of eigenvalues

acc_rate = no_accepted/iterations*100
S, iterations  = S[burn_in:-1], iterations-burn_in#, H[burn_in:-1]


# Plotting the action vs. Monte Carlo iterations
plt.plot(S)
plt.xlabel('Monte Carlo iterations')
plt.ylabel('Action <S>')
plt.grid(True)
plt.show()


print('Acceptance rate: ', acc_rate, '%')

eigen = np.ndarray.flatten(np.array(eigen))
plt.figure()
plt.hist(eigen, 60, density=True)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))