import numpy as np
G = 4*np.pi**2

#_________________________ FUNCTION __________________________________________#

def f(t, w, m):
    """ Derivative of the state vector w (which has 6 coordinates per element).
    Returns a derived state vector with the same shape as the input one. """

    w = w.reshape(len(m), 6)
    
    # Combined initial positions
    r = np.array([w[:,0], w[:,1], w[:,2]]).T # Start positions    
    
    # Loop to obtain the accelerations
    dvdt = np.zeros([len(m), 3])
    
    # Calculate the force of the sun
    sunforce = 0
    for j in range(1, len(m)):
        dxsun = r[0] - r[j] 
        dvssun = (m[j]*dxsun)/(np.linalg.norm(dxsun)**3)
        sunforce += dvssun # Perform the sums from the formula
    
    for i in range(len(r)): # For the element we're calculating the velocity
        dvs = np.zeros(len(r[0]))
        for j in range(len(r)): # The other elements
            if j != i: # To not pick the same i element          
                dx = r[i] - r[j] 
                inner = (m[j]*dx)/(np.linalg.norm(dx)**3)
                dvs += inner # Perform the sums from the formula
                
        dvdt[i] = -G*(dvs - sunforce) # Sum over all the values from dvs
        
    w_step1 = np.zeros([len(m), 6])    
    # Assign results from loop
    w_step1[:,0] = w[:,3] 
    w_step1[:,1] = w[:,4]  
    w_step1[:,2] = w[:,5]  
    w_step1[:,3] = dvdt[:,0]
    w_step1[:,4] = dvdt[:,1]
    w_step1[:,5] = dvdt[:,2]
    
    # Output vector in the same shape as the input on
    return w_step1.reshape(6*len(m))

def fast(t, w, m): 
    """ Derivative of the state vector w (which has 6 coordinates per element).
    Returns a derived state vector with the same shape as the input one. """
    
    n = len(m) # Number of elements
    # Reshape into matrix the input state vector
    w = w.reshape(n, 6)
    
    # Combined initial positions in a 4x3 array
    r = np.array([w[:,0], w[:,1], w[:,2]]).T # Start positions    

    # Stack n times r, reshape r and take difference to obtain dx
    rstack = r + np.zeros([n, n, 3]) 
    r_i = r.reshape([n, 1, 3])
    dx = r_i - rstack 
    
    # Need to mask dx to eliminate rows with zeros. Because diagonal behavior
    # on the first dimension of the 3D array we can use a diagonal mask
    mask = ~(np.eye(n, dtype='bool')) # Invert the diagonal mask
    dx_m = dx[mask].reshape(n, n-1, 3) 
    
    # Calculate the norm cubed, calculating the norm of each matrix
    norm = (np.linalg.norm(dx_m, axis=2).reshape(n, n-1, 1))**3 
    mass = ((m + np.zeros([n, n]))[mask]).reshape(n, n-1, 1) 
    mn = (mass/norm).reshape(n, n-1, 1)

    # Calculate forces
    forces = np.sum(-G*mn*dx_m, axis=1)
    fsun = forces[0]
    force = forces - fsun
    
    # Create array to store the values from the derivatives
    w_step = np.zeros([len(m), 6])    
    w_step[:,0] = w[:,3] 
    w_step[:,1] = w[:,4]  
    w_step[:,2] = w[:,5]  
    w_step[:,3] = force[:,0]
    w_step[:,4] = force[:,1]
    w_step[:,5] = force[:,2]
    
    return w_step.reshape(6*len(m))
        
#_________________________ INTEGRATOR ________________________________________#

def RK4(t, f, w, m, h):
    """ Runge Kutta 4th order integrator for a given (1 dimensional) 
    state vector w (with an horizontal shape), a time range from 0 
    to t and a time step h. """

    time = 0 # Initializes time
    n = len(m)   
    
    # Exctract coordinates from the state vectors
    xs = np.zeros([np.int(t/h)+1, n]) 
    ys = np.zeros([np.int(t/h)+1, n]) 
    zs = np.zeros([np.int(t/h)+1, n]) 
    
    # Reshape input state vector to obtain data of interest
    ws = w.reshape(n, 6) # len(m) is equal to the number of elements
    xs[0,:] = ws[:,0] # Assing starting values from the initial state vector
    ys[0,:] = ws[:,1] # to the resulting arrays as the first value
    zs[0,:] = ws[:,2] 
    
    i = 1 # Count start
    imax = int(t/h)+1 # Maximum step (defined by the time)

    while i<imax:
        
        fa = f(t, w, m) # Slope at start
        wb = w + 0.5*h*fa
        fb = f(t, wb, m) # Slope at midpoint 1
        wc = w + 0.5*h*fb
        fc = f(t, wc, m) # Slope at midpoint 2
        wd = w + h*fc
        fd = f(t, wd, m) # Slope at the end 
        
        # Weighted sum: update w
        w = w + (1/6)*h*fa + (1/3)*h*fb + (1/3)*h*fc + (1/6)*h*fd
        
        # Re-reshape in order to store variables
        ws = w.reshape(n, 6) 
        xs[i,:] = ws[:,0]
        ys[i,:] = ws[:,1]
        zs[i,:] = ws[:,2]
    
        i += 1
        time += h 
        
    return xs, ys, zs


