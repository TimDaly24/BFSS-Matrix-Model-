import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
start_time = datetime.now()


iterations = 10000
burn_in = int(iterations/10)
Lambda = 5 # Num of matrices
dim = 3 # number of dimensions, this is the i we sum over 
N = 7 # Size of matrices 
beta = 0.5
a = beta/Lambda
m = 1.0
eps = 0.06 # Raise this to lower the acceptance rate 


def cot(x):
    return (1/(np.tan(x)))


def Popov(theta):
    fp = 0
    for l in range(N):
        for m in range(N):
            if l != m:
                fp += np.log(np.abs(np.sin((theta[l] - theta[m]) / 2)))
    return -fp# Minus thing here

def gauge(theta):
    gauge_matrix = np.diag(np.exp(1j * theta))
    return gauge_matrix



def Hermitise(C):
    tri1 = np.triu(C) - np.diag(np.diag(C))
    tri2 = np.triu(C).T.conj()
    C = tri1 + tri2
    return C

def get_Hermitian(Lambda, N):
    path = []
    for _ in range(dim):
        matrices = []
        for _ in range(Lambda):
            D = np.diag(np.random.randn(N))
            C = np.random.normal(0, 1 / 2, size=(N, N))
            B = np.random.normal(0, 1 / 2, size=(N, N))
            B = 1j * (B - np.diag(B))
            C = C - np.diag(C)
            C = B + C + D
            C_sym = Hermitise(C)
            matrices.append(C_sym)
        path.append(matrices)
    return np.array(path)





def F(x,theta):
    D = gauge(theta)
    D_dagger = D.T.conj()
    
    F = []
    c = (2/a + a*m*m)
    F.append(1/a * (x[1] + D_dagger @ x[-1] @ D) - c*x[0]) # first term -dS/dx_1
        
    for t in range(1, Lambda-1):
        A = 1/a * (x[t-1] + x[t+1]) - c*x[t]
        F.append(A) # gathers all middle terms 
        
    F.append(1/a * (x[-2] + D @ x[0] @ D_dagger) - c*x[-1])
    
    return np.array(F)

def F_theta(theta,X):
    theta_mat = np.array[]
    for i in range(dim):
        temp = np.array[]
        for l in range(N):
            temp1 = 0
            temp2 = 0
            for m in range(N):
                temp1 += np.real(1j * (X[-1])[m,l] * (X[0])[l,m] * np.exp(1j * (theta[l] -theta[m])))
                
                if l != m:
                    temp2 += cot((theta[l] - theta[m])/2)
            temp.append(2/a * temp1 + temp2) # Possibly put temp1 and temp2 inside brackets 
        theta_mat += theta_mat + temp # questionable placement
    
    return np.array(theta_mat)


def Leap(X, P,Theta, P_Theta, space=eps): # Leapfrog integration
    x=X.copy()
    p=P.copy()
    phalf = np.zeros_like(x)

    
    theta = Theta.copy()
    p_theta = P_Theta.copy()
    p_theta_half = np.zeros_like(theta)

    for _ in range(10):
        phalf = p + space/2*F(x,theta)
        p_theta_half = p_theta + space/2*F_theta(theta, x)
            
        x = x + space*phalf
        theta = theta + space*p_theta_half
            
        p = phalf + space/2*F(x,theta)
        p_theta = p_theta_half + space/2*F_theta(theta, x)
        
    for j in range(Lambda):
        x[j] = (x[j]+p[j].conj().T)/2
        p[j] = (p[j]+p[j].conj().T)/2
            
        # Unsure if theta and p theta are matrices
        
    return x, p, theta, p_theta


def action(x,theta):
    Lambda = len(x)
    ting =0
    for i in range(dim):
        A1 = 0
        A2 = 0             # will use these to break up the calculation
        c = 2+a*a*m*m # defining this constant to be used later
        
        D = gauge(theta)
        D_dagger = D.T.conj()
        
        for i in range(Lambda-1):
            A1 += x[i]@x[i+1]
            A2 += np.trace(c*x[i]@x[i])
            
        A2 += np.trace(c*x[-1]@x[-1])
        A1 += x[-1] @ D @ x[0] @ D_dagger
        ting += np.real((-np.trace(A1) + 1/2 * A2)/a)
    return ting


def Ham(x, p, theta, p_theta):
    Lambda = len(x)
    n = len(x[0])
    misc=0
    for i in range(dim): 
        momenta = 0
        
        k = np.zeros((N,N), dtype=np.complex128)
        for t in range(Lambda):
            k += p[i][t]@p[i][t]
        
        for l in range(N):#Should be N?
            momenta += p_theta[i][l]**2
    
        kinetic = 0.5*np.trace(k) #over m ?     
        misc += kinetic + 1/2 * momenta + action(x,theta) + Popov(theta)
    return misc


def Metropolis(x,theta):
    x_prop1 = []
    p_prop1 = []
    theta_prop1 = []
    p_theta_prop1 = []
    acceptance = 0
    rand_p = get_Hermitian(Lambda, N)
    theta_kick = []
    for i in range(dim):
        theta_kick.append(np.array(np.random.normal(0, 1, size=(N))))
        
    for i in range(dim):
        x_prop, p_prop, theta_prop, p_theta_prop = Leap(x[i], rand_p[i],theta,theta_kick)
        x_prop1.append(x_prop)
        p_prop1.append(p_prop)
        theta_prop1.append(theta_prop)
        p_theta_prop1.append(p_theta_prop)

    Ham_current = Ham(x, rand_p, theta, theta_kick)
    Ham_prop = Ham(x_prop1, p_prop1, theta_prop1, p_theta_prop1)
    alpha = np.exp(Ham_current - Ham_prop)
    # alpha is probability of accepting new configuration if alpha<1
    if np.random.uniform(0,1) <= alpha:
        acceptance = 1
        return x_prop,theta_prop, acceptance	
    #decides whether or not to update the configuartion and tracks acceptance
    
    return x,theta, acceptance




x = get_Hermitian(Lambda, N)
theta = np.random.normal(0, 1, size=(N))# Possibly need to change this 
no_accepted=0
S = [action(x,theta)]
eigen = [np.linalg.eigvals(x)]
hamiltons = 0
ham_matrix = []
ham_actual = ((N**2)/2 + m/(np.exp(-beta) -1))#((N**2)/2 + (np.exp(-beta))/(1-np.exp(-beta))) 




thetas = []

def eig(x):
    return sum(np.ndarray.flatten(abs(np.linalg.eigvals(x))))

for i in range(1, iterations):
    # Propose new momenta from normal distribution
    x,theta, acceptance = Metropolis(x,theta)
    thetas.append(theta)
    no_accepted += acceptance
    S.append(action(x,theta))
    hamiltons = 0
    eigen.append(np.real(np.linalg.eigvals(x)))
    for i in range(dim):
        for j in range(Lambda):
            hamiltons += np.trace(x[i][j] @ x[i][j])
    ham_matrix.append(1 / (Lambda) * hamiltons)
    
acc_rate = no_accepted / iterations * 100
S, iterations = S[burn_in:-1], iterations-burn_in


# Plotting the action vs. Monte Carlo iterations
plt.plot(S)
plt.xlabel('Monte Carlo iterations')
plt.ylabel('Action <S>')
plt.axhline(y = (N*N * Lambda)/2, color="red")
plt.grid(True)
plt.show()

plt.plot(ham_matrix)
plt.xlabel('Monte Carlo iterations')
plt.ylabel('Hamiltonian <H>')
plt.axhline(y=ham_actual, color='orange')
plt.grid(True)
plt.show()

print('Acceptance rate: ', acc_rate, '%')
print('Expectation value of <S>: ', np.sum(S)/(iterations))
print('Expectation value should be: ', (N*N * Lambda) / 2)
print('Expectation value of <H>: ', np.real((np.sum(ham_matrix))) / iterations)
print('Expectation value should be: ', ham_actual)



eigen = np.ndarray.flatten(np.array(eigen))
plt.figure()
plt.hist(np.real(eigen), 40, density=True)
# do your work here
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))










