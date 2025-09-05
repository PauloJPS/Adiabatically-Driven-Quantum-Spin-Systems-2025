import qutip as qt
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick
import pickle

import matplotlib.ticker as ticker

from scipy.integrate import solve_ivp, odeint
from scipy.linalg import sqrtm
from scipy.integrate import simpson


plt.rcParams.update({
    "text.usetex": True,
    "font.family" : "lmodern", #   "font.serif": ["Times New Roman"],
    "font.weight": "bold",
    "text.latex.preamble": r"\usepackage[T1]{fontenc} \usepackage{lmodern} \usepackage{amsmath} \usepackage{mathptmx}\usepackage{amsfonts}",
    "xtick.minor.visible": True, 
    "ytick.minor.visible": True,
    
    "ytick.right": True,
    "ytick.left": True,

    "xtick.top": True,
    "xtick.bottom": True,
 
    #
    "xtick.direction": "in",
    "ytick.direction": "in",
    #
    "xtick.major.width": 1.5,     # major tick width in points
    "ytick.major.width": 1.5,     # major tick width in points
    #
    "xtick.minor.width": 1.5,     # minor tick width in points
    "ytick.minor.width": 1.5,     # minor tick width in points
    #
    "xtick.major.pad": 3.0,     # distance to major tick label in points
    "ytick.major.pad": 3.0,     # distance to major tick label in points
    #
    "xtick.minor.pad": 1.4,     # distance to the minor tick label in points
    "ytick.minor.pad": 1.4,     # distance to the minor tick label in points
    #
    "xtick.major.size": 5.5, 
    "ytick.major.size": 5.5,
    
    "xtick.minor.size": 3.5, 
    "ytick.minor.size": 3.5,
    #
    "xtick.labelsize": 16, 
    "ytick.labelsize": 16,
    #
    "legend.frameon": True, 
    "legend.fontsize": 20, 
    "legend.edgecolor": "white",
    "axes.titlesize": 20, 
    "axes.titleweight": "bold",
    "axes.labelsize":20
})

def GetOpenSystem(args):

    sigma_e = (qt.qeye(2) + qt.sigmaz())/2
    sigma_x = qt.sigmax()
      
    V = args["V"]/2
    gamma = args["gamma"] 
    Omega = args["Omega"]
    Delta = args["Delta"]
    alpha = args["alpha"]
    N = args["N"]
    
    ######

    # Single-qubit terms (e.g., sigma_e on each qubit)
    H_bare = 0
    for i in range(N):
        op_list = [qt.qeye(2) for _ in range(N)]
        op_list[i] = sigma_e 
        H_bare += Delta * qt.tensor(op_list)

    H_laser = 0
    for i in range(N):
        op_list = [qt.qeye(2) for _ in range(N)]
        op_list[i] = sigma_x
        H_laser += Omega * qt.tensor(op_list)

    H_detuning = 0
    for i in range(N):
        op_list = [qt.qeye(2) for _ in range(N)]
        op_list[i] = sigma_e
        H_detuning += Delta * qt.tensor(op_list)

    # Nearest-neighbor interaction terms 
    H_interaction = 0
    for i in range(N):
        for j in range(N):
            if i==j: continue
            else: 
                strength = V /np.abs(j-i)**alpha

                op_list_i = [qt.qeye(2) for _ in range(N)]
                op_list_j = [qt.qeye(2) for _ in range(N)]

                op_list_i[i] = sigma_e
                op_list_j[j] = sigma_e

                op_list_i = qt.tensor(op_list_i)
                op_list_j = qt.tensor(op_list_j)

                H_interaction += strength * op_list_i * op_list_j

    # Jump operators 
    c_list = []
    for i in range(N):
        op_list = [qt.qeye(2) for _ in range(N)]
        op_list[i] = sigma_e
        c_list.append(np.sqrt(gamma) * qt.tensor(op_list))

    return [H_bare, H_laser, H_detuning, H_interaction], c_list

###############

def GetPsi0(args, up=True):
    
    N = args["N"]
    if up == True: psi0 = qt.tensor([qt.basis(2,0) for _ in range(N)])
    else: psi0 = qt.tensor([qt.basis(2,1) for _ in range(N)])

    return psi0/psi0.norm()

def get_sigma_lists(N):
    """
        Return lists of sigma_x and sigma_y tensors for N spins
    """
    sigma_x_list = []
    sigma_y_list = []
    for i in range(N):
        ops_x = [qt.qeye(2) if j != i else qt.sigmax() for j in range(N)]
        ops_y = [qt.qeye(2) if j != i else qt.sigmay() for j in range(N)]
        sigma_x_list.append(qt.tensor(ops_x))
        sigma_y_list.append(qt.tensor(ops_y))
    return sigma_x_list, sigma_y_list

def get_sigma_e_lists(N):
    """
        Return lists of sigma_x and sigma_y tensors for N spins
    """
    sigma_e_list = []
    for i in range(N):
        ops_e = [ qt.qeye(2) if j != i else  (qt.sigmaz() + qt.qeye(2))/2 for j in range(N)]
        sigma_e_list.append(qt.tensor(ops_e))
    return sigma_e_list



################
################
################


def Interpolation(t, args):
    return (4*np.pi/args["tau"]) * np.sin(np.pi*t/args["tau"])**2

def InterpolationSquare_integration(t, args):
    return (np.pi/( 2 * args["tau"]**2)
                * (args["tau"]*np.sin(4*np.pi*t/args["tau"]) + 12*np.pi*t - 8*args["tau"]*np.sin(2*np.pi*t/args["tau"])))

def Operator_valued_rates(k, args):

    # Nearest-neighbor interaction terms 
    C_k = 0 
    for i in range(args["N"]):
        if i!=k:
            op_list = [qt.qeye(2) for _ in range(args["N"])]
            op_list[i] = (qt.qeye(2) + qt.sigmaz())/2
            op_list = qt.tensor(op_list)

            strength = args["V"] * np.abs(k-i, dtype=float)**(-args["alpha"])
            C_k += strength * op_list
        else: continue
    C_k += args["Delta"]              
    Lambda_k = (args["gamma"]**2/4 + C_k.copy()**2).inv()
    
    return Lambda_k, C_k


################
################
################

def Liuvillian_QuTip(args): 
    """
        Calcualte the Hamiltonian and jump operators for the adiabatic propagator
    """
    ############## Projectors 
    sigma_x_list, sigma_y_list =  get_sigma_lists(args["N"])

    Hamiltonian = 0 
    JumpOp_list = []

    for k in range(args["N"]): # Goes through all the spins 
        Lambda_k, C_k = Operator_valued_rates(k, args)

        #### 
        Heff_x = args["gamma"]/2 * sigma_x_list[k] 
        Heff_y = C_k * sigma_y_list[k]
        Hamiltonian += Lambda_k * (Heff_x + Heff_y) 

        JumpOp_list.append(Lambda_k.sqrtm() * sigma_x_list[k]) 
        #### 

    return Hamiltonian, JumpOp_list

def Adiabatic_Propagator(rho0, tf, args):

    """
        Calcualte the time-evolution in the Schrödinger picture for the final time tf 
    """
    ############## Projectors 
    sigma_x_list, sigma_y_list =  get_sigma_lists(args["N"])
    ############## Time-dependent parameters 

    Omega_t = Interpolation(tf, args)
    omega2_t = InterpolationSquare_integration(tf, args) 
    ############# 
    
    FirstOrder = 0 
    for k in range(args["N"]): # Goes through all the spins 
        Lambda_k, C_k = Operator_valued_rates(k, args)
        #### 
        Heff_x = args["gamma"]/2 * (Lambda_k * sigma_x_list[k])
        Heff_y = Lambda_k * C_k * sigma_y_list[k] 

        #### 
        FirstOrder += -1.j * Omega_t * ((Heff_x + Heff_y) * rho0 - rho0 * (Heff_x + Heff_y)) # Hamiltonian 
        FirstOrder += omega2_t * Lambda_k * (sigma_x_list[k] * rho0 * sigma_x_list[k] - rho0) # Dissipation

    return rho0 + FirstOrder 

def TimeDependent_Dynamics(rho0, tf, args, only_final=False):

    """ 
        Calculate the exact dynamics 
    """ 

    options = {"atol":1e-12, "rtol":1e-12}

    tlist = np.linspace(0, tf, 1001)

    H, c_list = GetOpenSystem(args)
    H_bare, H_laser, H_detuning, H_interaction  = H

    H = [H_bare, [H_laser, Interpolation], H_detuning, H_interaction]

    if only_final==True:
        options.update({"store_final_state":True})
        options.update({"store_states":False})
        result = qt.mesolve(H, rho0, tlist, c_ops=c_list, args=args, options=options)
        return result.final_state
    else: return qt.mesolve(H, rho0, tlist, c_ops=c_list, args=args, options=options)

###################
###################
###################


def GetParameters():

    V = 1
    alpha = 3
    gamma = 1.0
    Omega = 1.0
    Delta = 0.0
    tau = 1.0
    N = 2
   
    args = {"V":V,
            "N":N,
            "gamma":gamma,
            "Omega":Omega,
            "Delta":Delta,
            "alpha":alpha,
            "tau":tau}

    return args 

################
################
################

