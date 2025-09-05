import qutip as qt
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick
import pickle
import sys
import matplotlib.ticker as ticker
import time

from scipy.integrate import solve_ivp, odeint 
from scipy.linalg import sqrtm
from scipy.integrate import simpson

# adding Folder_2/subfolder to the system path
sys.path.insert(0, './')

from main import *

def Adiabatic_Error():
    args = GetParameters()
    
    args["V"] = 3 
    args["gamma"] = 1.0
    N_list = [3, 4, 5]
    tau_list = np.logspace(np.log(0.1), np.log(50000), 300, base=np.exp(1)) 

    Fidelity_Expon_local= [[] for _ in N_list]
    Fidelity_Expon_global = [[] for _ in N_list]

    Fidelity_Expon_local_half= [[] for _ in N_list]
    Fidelity_Expon_global_half = [[] for _ in N_list]

    #### 

    for n, N in enumerate(N_list):

        print(f"N = {N}\n")
        args["N"] = N

        #### Adiabatic operators 
        H, c_ops = Liuvillian_QuTip(args)

        acc = 0 
        for tau in tau_list:
            t0 = time.time()
            args["tau"] = tau
            Omega_t = Interpolation(tau, args.copy())
            omega2_t = InterpolationSquare_integration(tau, args.copy())

            Omega_t_half = Interpolation(tau/2, args.copy())
            omega2_t_half = InterpolationSquare_integration(tau/2, args.copy())

            ###
            psi0 = GetPsi0(args.copy(), up=False)
            Pp = psi0 * psi0.dag()
            ### 

            rho_Exact_half = TimeDependent_Dynamics(Pp.copy(), tau/2, args.copy(), only_final=True)
            rho_Exact = TimeDependent_Dynamics(Pp.copy(), tau, args.copy(), only_final=True)

            ### Get Lindbladian  
            Propagator_half = qt.liouvillian(Omega_t_half*H, c_ops=[np.sqrt(omega2_t_half)*c for c in c_ops])
            Propagator = qt.liouvillian(Omega_t*H, c_ops=[np.sqrt(omega2_t)*c for c in c_ops])
            
            ### get states 
            rho_Expon_half = qt.vector_to_operator(Propagator_half.expm() * qt.operator_to_vector(Pp.copy()))
            rho_Expon = qt.vector_to_operator(Propagator.expm() * qt.operator_to_vector(Pp.copy()))

            ###
            Fidelity_Expon_local[n].append([qt.tracedist(qt.ptrace(rho_Exact,0),qt.ptrace(rho_Expon,0))])
            Fidelity_Expon_global[n].append([qt.tracedist(rho_Exact, rho_Expon)])

            Fidelity_Expon_local_half[n].append([qt.tracedist(qt.ptrace(rho_Exact_half,0),qt.ptrace(rho_Expon_half,0))])
            Fidelity_Expon_global_half[n].append([qt.tracedist(rho_Exact_half, rho_Expon_half)])

            
            acc += time.time() - t0
            print(f"{tau} -- {time.time()-t0} -- acc = {acc}\n")

    np.save("Data/Fidelity_Expon_local", Fidelity_Expon_local)
    np.save("Data/Fidelity_Expon_global", Fidelity_Expon_global)
    np.save("Data/Fidelity_Expon_local_half", Fidelity_Expon_local_half)
    np.save("Data/Fidelity_Expon_global_half", Fidelity_Expon_global_half)

    np.save("Data/tau_list_local", tau_list)

    return tau_list, Fidelity_Expon_local, Fidelity_Expon_global, Fidelity_Expon_local_half, Fidelity_Expon_global_half


