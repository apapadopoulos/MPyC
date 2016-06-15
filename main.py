#!/usr/bin/python
import numpy as np
import argparse
import os
import sys

# My libraries
import libs.mpyc as reg
import libs.utils as ut
from libs.init_mpc import initialize_mpc

def main():
    ## Manage command line inputs
    # Defining command line options to find out the algorithm
    parser = argparse.ArgumentParser( \
        description='Run MPC simulator.', \
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--simTime',
        type = int,
        help = 'Simulation time.',
        default = 500)

    parser.add_argument('--predictionHorizon',
        type = int,
        help = 'Prediction horizon.',
        default = 5)

    parser.add_argument('--wo',
        type = float,
        help = 'Output weights.',
        default = 1.0)

    parser.add_argument('--wdu',
        type = float,
        help = 'Delta U weights.',
        default = 1.0)

    parser.add_argument('--var_noise',
        type = float,
        help = 'Noise acting on the system output.',
        default = 0.1)

    parser.add_argument('--optim',
        type = int,
        help = 'Online optimization.',
        default = 1)

    parser.add_argument('--fast',
        type = int,
        help = 'Fast version.',
        default = 0)

    parser.add_argument('--timeVarying',
        type = int,
        help = 'Time-varying Kalman filter.',
        default = 0)

    # Parsing the command line inputs
    args = parser.parse_args()


    # Initialize matrices for MPC
    A,B,C,D,Umin,Umax,DeltaUmin,DeltaUmax,Qn,Rn,Lk,Pk,sp = initialize_mpc();
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    # Control parameters
    L = args.predictionHorizon # Control horizon
    Q = args.wo * np.eye(L*p)  # Weighting matrix on the error
    R = args.wdu * np.eye(L*m) # Weighting matrix on the Delta Control

    # Regulator
    mpc = reg.MPCController(A,B,C,D,\
                            L,Q,R,\
                            Lk,Pk,Qn,Rn,\
                            Umin,Umax,DeltaUmin,DeltaUmax,\
                            optim=bool(args.optim),\
                            fast=bool(args.fast),\
                            time_varying=bool(args.timeVarying))

    ## Simulation loop
    Tfin = args.simTime
    var_noise = args.var_noise

    # Vectors for storing the result
    x  = np.zeros((n,Tfin))
    y  = np.zeros((p,Tfin))
    u  = np.zeros((m,Tfin))
    yo = np.zeros((p,Tfin))

    # Auxiliary vectors
    xx  = np.zeros((n,1))
    yy  = np.zeros((p,1))
    uu  = np.zeros((m,1))

    for kk in range(0,Tfin):
        ut.progress(kk,Tfin)
        # Setpoint variation
        if kk < np.floor(Tfin/3):
            sp = np.matrix([[5],[2]])
        else:
            sp = np.matrix([[8],[1]])
        if kk > np.floor(0.9*Tfin):
            sp = np.matrix([[5],[2]])

        # Control law
        uu = mpc.compute_u(yy,sp)

        # System update
        # if kk < np.floor(Tfin/4) and kk > np.floor(3*Tfin/4):
        #     yy = Cs*xx + var_noise*np.random.rand(p,1)
        #     xx = As*xx + Bs*uu
        # else:
        #     yy = C*xx + var_noise*np.random.rand(p,1)
        #     xx = A*xx + B*uu
        yy = C*xx + var_noise*np.random.rand(p,1)
        xx = A*xx + B*uu

        # Disturbances acting on the system
        if kk > np.floor(Tfin/2): # Load disturbances
            xx = xx + 0.2*np.ones((n,1));
        if kk> np.floor(2*Tfin/3): # Output disturbances
            yy = yy + 1*np.ones((p,1));

        # Saving results for plotting
        yo[:,kk] = sp.T
        y[:,kk] = yy.T
        u[:,kk] = uu.T
        if kk+1 < Tfin:
            x[:,kk+1] = xx.T

    print '\nSimulation completed!\n'
    
    # Plotting results
    time = np.matrix(range(0,Tfin))
    ut.plot_res(time,x,y,u,yo)



if __name__ == "__main__":
    sys.exit(main())



