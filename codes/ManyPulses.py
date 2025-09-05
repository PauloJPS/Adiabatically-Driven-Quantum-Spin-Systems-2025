import qutip as qt
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick
import pickle
import sys
import matplotlib.ticker as ticker

from scipy.integrate import solve_ivp, odeint 
from scipy.linalg import sqrtm
from scipy.integrate import simpson

# adding Folder_2/subfolder to the system path
sys.path.insert(0, './')

from main import *


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

def NPulses_dynamics(tau, N_pulses):

    args = GetParameters()
    N = 2
    args["V"] = 3.0
    args["gamma"] = 1.0
    args["N"] = N
    args["tau"] = tau
    #### 

    H, c_ops = Liuvillian_QuTip(args)
    Omega_t = Interpolation(tau, args.copy())
    omega2_t= InterpolationSquare_integration(tau, args.copy())

    Propagator = qt.liouvillian(Omega_t*H, c_ops=[np.sqrt(omega2_t)*c for c in c_ops])

    psi0 = GetPsi0(args.copy(), up=False)
    Pp = psi0 * psi0.dag()
    
    rho_Exact = [Pp]
    rho_First = [Pp]
    
    for n in range(N_pulses):
        #### Adiabatic operators 

        ### 
        rho_Exact.append(TimeDependent_Dynamics(rho_Exact[-1].copy(), tau, args.copy(), only_final=True))

        ###
        rho_First.append(qt.vector_to_operator((1 + Propagator)  * qt.operator_to_vector(rho_First[-1].copy())))

    return rho_Exact, rho_First

def KC_dynamics(tau, N_pulses):

    args = GetParameters()
    N = 4
    args["gamma"] = 1.0
    args["N"] = N
    args["tau"] = tau
    #### 

    Omega_t = Interpolation(tau, args.copy())
    omega2_t= InterpolationSquare_integration(tau, args.copy())


    sigma_e_list = get_sigma_e_lists(args["N"])
    Se = sum(sigma_e_list)

    psi0 = GetPsi0(args.copy(), up=False)
    Pp = psi0 * psi0.dag()
    
    Se_Exact = [[0], [0]]
    Se_Expon = [[0], [0]]
    Se_First = [[0], [0]]

    Vlist = [0, 5]
    
    for i, V in enumerate(Vlist): 
        rho_Exact = [Pp.copy()]
        rho_Expon = [Pp.copy()]
        rho_First = [Pp.copy()]
        args["V"] = V

        H, c_ops = Liuvillian_QuTip(args)
        Propagator = qt.liouvillian(Omega_t*H, c_ops=[np.sqrt(omega2_t)*c for c in c_ops])
        for n in range(N_pulses):
            #### Adiabatic operators 

            rho_Exact.append(TimeDependent_Dynamics(rho_Exact[-1].copy(), tau, args.copy(), only_final=True))
            Se_Exact[i].append(qt.expect(Se, rho_Exact[-1]))

            ####
            rho_First.append(qt.vector_to_operator((1 + Propagator)  * qt.operator_to_vector(rho_First[-1].copy())))
            Se_First[i].append(qt.expect(Se, rho_First[-1]))

            rho_Expon.append(qt.vector_to_operator(Propagator.expm() * qt.operator_to_vector(rho_Expon[-1].copy())))
            Se_Expon[i].append(qt.expect(Se, rho_Expon[-1]))
 
    np.save("Data/Density_exact", Se_Exact)
    np.save("Data/Density_expon", Se_Expon)
    np.save("Data/Density_first", Se_First)

    return np.array(Se_Exact)/N, np.array(Se_Expon)/N, np.array(Se_First)/N


def KC_dynamics_coherences(tau, N_pulses):

    args = GetParameters()
    N = 4
    args["gamma"] = 1.0
    args["N"] = N
    args["tau"] = tau
    #### 

    Omega_t = Interpolation(tau, args.copy())
    omega2_t= InterpolationSquare_integration(tau, args.copy())
    Omega_t_Half = Interpolation(tau/2, args.copy())
    omega2_t_Half= InterpolationSquare_integration(tau/2, args.copy())



    sigma_e_list = get_sigma_e_lists(args["N"])
    Se = sum(sigma_e_list)

    psi0 = GetPsi0(args.copy(), up=False)
    Pp = psi0 * psi0.dag()
    
    C_Exact = [[], []]
    C_Expon = [[], []]
    C_First = [[], []]

    Vlist = [0, 5]
    
    for i, V in enumerate(Vlist): 
        rho_Exact = [Pp.copy()]
        rho_Expon = [Pp.copy()]
        rho_First = [Pp.copy()]
        args["V"] = V

        H, c_ops = Liuvillian_QuTip(args)
        Propagator = qt.liouvillian(Omega_t*H, c_ops=[np.sqrt(omega2_t)*c for c in c_ops])
        Propagator_Half = qt.liouvillian(Omega_t_Half*H, c_ops=[np.sqrt(omega2_t_Half)*c for c in c_ops])
        for n in range(N_pulses):
             
            aux_Exact = TimeDependent_Dynamics(rho_Exact[-1].copy(), tau/2, args.copy(), only_final=True)
            C_Exact[i].append(qt.entropy_vn(qt.Qobj(np.diag(aux_Exact.diag()), dims=aux_Exact.dims)) 
                            - qt.entropy_vn(aux_Exact))

            rho_Exact.append(TimeDependent_Dynamics(rho_Exact[-1].copy(), tau, args.copy(), only_final=True))

            ####
            aux_First = qt.vector_to_operator((1 + Propagator_Half)  * qt.operator_to_vector(rho_First[-1].copy()))
            C_First[i].append(qt.entropy_vn(qt.Qobj(np.diag(aux_First.diag()), dims=aux_First.dims)) 
                            - qt.entropy_vn(aux_First))

            rho_First.append(qt.vector_to_operator((1 + Propagator)  * qt.operator_to_vector(rho_First[-1].copy())))
            
            ####
            aux_Expon = qt.vector_to_operator(Propagator_Half.expm()  * qt.operator_to_vector(rho_Expon[-1].copy()))
            C_Expon[i].append(qt.entropy_vn(qt.Qobj(np.diag(aux_Expon.diag()), dims=aux_Expon.dims)) 
                            - qt.entropy_vn(aux_Expon))

            rho_Expon.append(qt.vector_to_operator(Propagator.expm() * qt.operator_to_vector(rho_Expon[-1].copy())))

    
    np.save("Data/Coherence_exact", C_Exact)
    np.save("Data/Coherence_expon", C_Expon)
    np.save("Data/Coherence_first", C_First)

    return np.array(C_Exact), np.array(C_Expon), np.array(C_First)

