import numpy as np;

def initialize_mpc():
    # Model
    A = np.matrix([[0.5, 1.0, 0.0],[0.0, 0.4, 2.0],[0.0, 0.0,-0.1]])
    B = np.matrix([[1.0,2.0,0.0],[2.0,1.0,1.0],[0.5,0.0,0.0]])
    C = np.matrix([[ 1.0, 2.0, 0.0],[0.0, 1.0, 0.0 ]]);

    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    D = np.zeros((p,m))

    DeltaUmin = -0.1*np.ones((m,1))
    DeltaUmax = 0.05*np.ones((m,1))
    Umin = -1*np.ones((m,1))
    Umax = 1*np.ones((m,1))

    # Kalman filter parameters
    Qn = np.eye(m);
    Rn = np.eye(p);
    Lk = np.matrix([[0.44826,0.117478],[0.142742,0.277767],[-0.00243795,-0.00717033]])
    Pk = np.matrix([[5.22413,4.07137,0.498781],[4.07137,6.65598,0.972756],[0.498781,0.972756,0.251219]])

    # Setpoint
    sp = np.matrix([[10],[1]]);

    return A,B,C,D,Umin,Umax,DeltaUmin,DeltaUmax,Qn,Rn,Lk,Pk,sp;
