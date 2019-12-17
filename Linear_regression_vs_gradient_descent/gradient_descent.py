"""gradient_descent - my module"""
import numpy as np
def myDescent(x, y,learn_rate, conv_threshold,batch_size,max_iter):
    """
    myDescent(x, y, learn_rate, conv_threshold, batch_size, max_iter) parameters:
 - x: Independent variable
 - y: Dependent variable
 - learn_rate: the rate which gradients are updated; low causes slow convergence and high causes divergence and missing the global minimum
 - conv_threshold: the difference between the old MSE(Mean Square Error) and new MSE on each iteration
 - batch_size: number of observations considered at each iteration fo updating gradients; high number causes lower iterations, and lower number causes decrease in errors. Ideally the number should be a value of 30 due to statistical significance.
 - max_iter: maximum number of iteration, beyond which the algorithm will be stoped.
    
    """
    converged = False
    iter = 0
    m = batch_size
    t0 = np.random.random(x.shape[1])
    t1 = np.random.random(x.shape[1])
    MSE = (sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)])/ m)    

    while not converged:        
        grad0 = 1.0/m * sum([(t0 + t1*x[i] - y[i]) for i in range(m)]) 
        grad1 = 1.0/m * sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(m)])
        temp0 = t0 - learn_rate * grad0
        temp1 = t1 - learn_rate * grad1
        t0 = temp0
        t1 = temp1
        MSE_New = (sum( [ (t0 + t1*x[i] - y[i])**2 for i in range(m)] ) / m)

        if abs(MSE - MSE_New ) <= conv_threshold:
            print ('Converged, iterations: ', iter)
            converged = True
    
        MSE = MSE_New   
        iter += 1 
    
        if iter == max_iter:
            print ('Max interactions reached')
            converged = True

    return t0,t1