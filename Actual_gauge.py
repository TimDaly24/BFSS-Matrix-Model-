import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
start_time = datetime.now()
plt.style.use('bmh')
bleu = (52/255, 207/255, 235/255)  
greeen = (165/255, 235/255, 52/255)


iterations = 10000
burn_in = int(iterations/10)
Lambda = 5
N = 25 # Size of matrices 
beta = 0.3
a = beta/Lambda
m = 1.0
eps = 0.037 # Raise this to lower the acceptance rate 


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

def get_Hermitian(Lambda,N):
    path = [] 
    for _ in range(Lambda):
        D = np.diag(np.random.randn(N))
        C = np.random.normal(0, 1/2, size=(N, N))
        B = np.random.normal(0, 1/2, size=(N, N))
        B = 1j*(B - np.diag(B))
        C = C - np.diag(C)
        C = B+C+D
        C_sym = Hermitise(C)
        path.append(C_sym)
    return path




def F(x,theta):
    D = gauge(theta)
    D_dagger = D.T.conj()
    
    c = (2/a + a*m*m)
    F = [1/a * (x[1] + D_dagger @ x[-1] @ D) - c*x[0]] # first term -dS/dx_1
    
    
    Lambda = len(x)
    for i in range(1, Lambda-1):
        A = 1/a * (x[i-1] + x[i+1]) - c*x[i]
        F.append(A) # gathers all middle terms 
    
    F.append(1/a * (x[-2] + D @ x[0] @ D_dagger) - c*x[-1])
    
    return np.array(F)

def F_theta(theta,X):
    theta_mat = []

    for l in range(N):
        temp1 = 0
        temp2 = 0
        for m in range(N):
            temp1 += np.real(1j * (X[-1])[m,l] * (X[0])[l,m] * np.exp(1j * (theta[l] -theta[m])))
        
            if l != m:
                temp2 += cot((theta[l] - theta[m])/2)
        theta_mat.append(2/a * temp1 + temp2) # Possibly put temp1 and temp2 inside brackets 
    
    return np.array(theta_mat)


def Leap(X, P,Theta, P_Theta, space=eps): # Leapfrog integration
    x=X.copy()
    p=P.copy()
    phalf = np.zeros_like(x)

    
    theta = Theta.copy()
    p_theta = P_Theta.copy()
    p_theta_half = np.zeros_like(theta)
    
    # Maybe add in copies for theta
    for i in range(5):
        phalf = p + space/2*F(x,theta)
        p_theta_half = p_theta + space/2*F_theta(theta, x)
        
        x = x + space*phalf
        theta = theta + space*p_theta_half
        
        p = phalf + space/2*F(x,theta)
        p_theta = p_theta_half + space/2*F_theta(theta, x)
        
    for i in range(Lambda):
        x[i] = (x[i]+x[i].conj().T)/2
        p[i] = (p[i]+p[i].conj().T)/2
        
        # Unsure if theta and p theta are matrices
        
    return x, p, theta, p_theta


def action(x,theta):
    Lambda = len(x)
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
    # derived from non-gauged version so g&g_inv may be in wrong order here, I guessed it
    
    return np.real((-np.trace(A1) + 1/2 * A2)/a)


def Ham(x, p, theta, p_theta):
    Lambda = len(x)
    n = len(x[0])
    momenta = 0
    k = np.zeros((N,N), dtype=np.complex128)
    for i in range(Lambda):
        k += p[i]@p[i]
        
    for l in range(N):#Should be N?
        momenta += p_theta[l]**2
    
    kinetic = 0.5*np.trace(k) #over m ?     
    misc = kinetic + 1/2 * momenta + action(x,theta) + Popov(theta)
    return misc


def Metropolis(x,theta):

    acceptance = 0
    rand_p = get_Hermitian(Lambda, N)
    theta_kick = np.array(np.random.normal(0, 1, size=(N)))
    x_prop, p_prop, theta_prop, p_theta_prop = Leap(x, rand_p,theta,theta_kick)
    Ham_current = Ham(x, rand_p, theta, theta_kick)
    Ham_prop = Ham(x_prop, p_prop, theta_prop, p_theta_prop)
    alpha = np.exp(Ham_current - Ham_prop)
    # alpha is probability of accepting new configuration if alpha<1
    if np.random.uniform(0,1) <= alpha:
        acceptance = 1
        return x_prop,theta_prop,Ham_prop,acceptance	
    #decides whether or not to update the configuartion and tracks acceptance
    
    return x,theta,Ham_current, acceptance




x = (get_Hermitian(Lambda, N))
for i in range(Lambda):
    x[i]=x[i]*1/5
theta = np.random.normal(0, 1, size=(N))# Possibly need to change this 
no_accepted=0
S = [action(x,theta)]
eigen = [np.linalg.eigvals(x)]
hamiltons = 0
ting = []
ham_matrix = []
ham_actual = ((N**2)/2 + m/(np.exp(-beta) -1))#((N**2)/2 + (np.exp(-beta))/(1-np.exp(-beta))) 




thetas = []

def eig(x):
    return sum(np.ndarray.flatten(abs(np.linalg.eigvals(x))))

for i in range(1, iterations):
    # Propose new momenta from normal distribution
    x,theta,H, acceptance = Metropolis(x,theta)
    thetas.append(theta)
    no_accepted += acceptance
    S.append(action(x,theta))
    ting.append(H)
    hamiltons = 0
    '''eigen.append(np.real(np.linalg.eigvals(x)))
    for j in range(Lambda):
        hamiltons += np.trace(x[j] @ x[j])
    ham_matrix.append(1 / (Lambda) * hamiltons)'''
    
acc_rate = no_accepted / iterations * 100
S, iterations = S[burn_in:-1], iterations-burn_in


# Plotting the action vs. Monte Carlo iterations
plt.plot(S,label ='Action')
plt.xlabel('Monte Carlo iterations')
plt.ylabel('<S>')
plt.axhline(y = (N*N * Lambda)/2,color="orange",label='Analytic')
plt.legend()
plt.grid(True)
plt.show()


'''plt.plot(ham_matrix)
plt.xlabel('Monte Carlo iterations')
plt.ylabel('Hamiltonian <H>')
plt.axhline(y=ham_actual, color='orange')
plt.grid(True)
plt.show()'''

print('Acceptance rate: ', acc_rate, '%')
print('Expectation value of <S>: ', np.sum(S)/(iterations))
print('Expectation value should be: ', (N*N * Lambda) / 2)
#print('Expectation value of <H>: ',np.real((np.sum(ham_matrix))) / iterations)# np.real((np.sum(ham_matrix))) / iterations)
#print('Expectation value should be: ', ham_actual)


'''
eigen = np.ndarray.flatten(np.array(eigen))
plt.figure()
plt.hist(np.real(eigen), 30, density=True)'''
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

