''' Strategy for no-GPS zones
We maintain a probability distribution Q across all robot positions (Markov localization)
We use only one GPS measurement to initialize this distribution, then we perform updates at each step without GPS

Say we want to compute Q' the updated probability distribution and let x be the estimated current position
We use Metropolis algorithm to compute Q' over the neighbors of x
=> we don't update entire matrix (reduces cost of the algorithm)
'''
import random
import numpy as np

class MetropolisAlgorithm:
    '''Metropolis algorithm used to update belief distribution at each step'''

    #Metropolis-Hastings ratio
    def r(self, x, y):
        return x+y

    #Acceptance rate using Metropolis rule
    def metropolisacceptance(self, x, y):
        return x+y
    
    #Estimate sample of unknown distribution Belief(t + 1)
    def update(self, curPosition, Niter):
        Xn = curPosition
        while(Niter > 0):
            candidate = (random.uniform(0,1), random.uniform(0,1))
            u = random.uniform(0,1)
            if( u < self.metropolisacceptance(Xn, candidate)): #accepter proposition locale
                Xn = candidate
            #else refuse candidate => X_{n+1} = X_n
            Niter -= 1
        return Xn

class MarkovLocalization:
    '''Keeps track of drone position in no-GPS zones'''

    def __init__(self, N, gps_x, gps_y):
        self.N = N #we use an NxN state matrix
        self.beliefDistribution = np.zeros((N,N))
        self.beliefDistribution[gps_x][gps_y] = 1 #we use gps to initialize the distribution

    def update_belief_distribution(self):
        return self.beliefDistribution
    
    def estimate_position(self):
        '''Compute expected value of position using belief distribution'''
        expected_x = 0
        expected_y = 0
        for i in range(self.N):
            for j in range(self.N):
                expected_x += i*self.beliefDistribution[i][j]
                expected_y += j*self.beliefDistribution[i][j]
        return (expected_x, expected_y)
