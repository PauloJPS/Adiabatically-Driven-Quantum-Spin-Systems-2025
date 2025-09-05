import qutip as qt
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick
import pickle
import sys
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from scipy.integrate import solve_ivp, odeint
from scipy.linalg import sqrtm
from scipy.integrate import simpson


# adding Folder_2/subfolder to the system path
sys.path.insert(0, './')

from main import *


plt.rcParams.update({
    "figure.dpi": 150,  # Higher resolution for screen and print
    "savefig.dpi": 300,

    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

    "text.latex.preamble": r"""
        \usepackage[T1]{fontenc}
        \usepackage{mathptmx}
        \usepackage{amsmath}
        \usepackage{amsfonts}
    """,

    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.minor.size": 1,
    "ytick.minor.size": 1,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": True,
    "ytick.right": True,

    "legend.frameon": True,
    "legend.edgecolor": "white",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def PlotFig2():

    #### getting simple data 
    args = GetParameters() 
    args["tau"] = 400
    args["N"] = 2
    args["V"] = 1
    
    tlist = np.linspace(0, args["tau"], 1001)
    psi0 = GetPsi0(args) 
    Pp = psi0 * psi0.dag() 

    H, c_ops = Liuvillian_QuTip(args)   

    exact = TimeDependent_Dynamics(Pp, args["tau"], args).states
    appro = [] 
    
    for tf in tlist: 
        Omega_t = Interpolation(tf, args.copy())
        omega2_t = InterpolationSquare_integration(tf, args.copy())
        Propagator = qt.liouvillian(Omega_t*H, c_ops=[np.sqrt(omega2_t)*c for c in c_ops])
        appro.append(qt.vector_to_operator(Propagator.expm() * qt.operator_to_vector(Pp.copy())))

    sigma_x_list, sigma_y_list = get_sigma_lists(args["N"])
    Sx = sum(sigma_x_list)
    Sy = sum(sigma_y_list)

    #### loading data 

    F_exp_l = np.load("Data/Fidelity_Expon_local.npy")
    F_exp_g = np.load("Data/Fidelity_Expon_global.npy")
    F_exp_gh = np.load("Data/Fidelity_Expon_global_half.npy")

    tau_list = np.load("Data/tau_list_local.npy")
    tau_list_ = np.load("Data/tau_list_.npy")


    #F_exp = np.load("Data/Fidelity_Expon.npy")
    #F_fir = np.load("Data/Fidelity_First.npy")
    #F_fif = np.load("Data/Fidelity_fifth.npy")

    F_exi = np.load("Data/Fidelity_Exact_Infin.npy")
    F_epi = np.load("Data/Fidelity_Expon_Infin.npy")
   
    #### starting plotting 
    fig = plt.figure(figsize=(5.5, 5))
   
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    gs_bottom_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], hspace=0.0)

    # Bottom-right top (C1)

    ax1 = fig.add_subplot(gs_bottom_right[0, 0], sharex=None)
    ax2 = fig.add_subplot(gs_bottom_right[1, 0], sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)



    # Plotting example content
    
    ax1.plot(tlist/args["tau"], [rho[3][3] for rho in exact], color="black") 
    ax1.plot(tlist/args["tau"], [rho[2][2] for rho in exact], color="black") 
    ax1.plot(tlist/args["tau"], [rho[1][1] for rho in exact], color="black") 
    ax1.plot(tlist/args["tau"], [rho[0][0] for rho in exact], color="black") 

    ax1.plot(tlist/args["tau"], [rho[3][3] for rho in appro], label=r"$p_{00}$", color="#e41a1c", linestyle="--") 
    ax1.plot(tlist/args["tau"], [rho[2][2] for rho in appro], label=r"$p_{01}$", color="#377eb8", linestyle="--") 
    ax1.plot(tlist/args["tau"], [rho[0][0] for rho in appro], label=r"$p_{11}$", color="#4daf4a", linestyle="--") 

    ax1.set_xlim((0, 1))  # Hide x

 
    ax1.legend(edgecolor="black", framealpha=1, 
                        handlelength=1, borderpad=0.3, fontsize=12, loc=0, labelspacing=0.0) 


    ####################### 

    ax2.plot(tlist/args["tau"], [qt.expect(Sx, rho) for rho in exact], color="black") 
    ax2.plot(tlist/args["tau"], [qt.expect(Sy, rho) for rho in exact], color="black") 
                                                                            
    ax2.plot(tlist/args["tau"], [qt.expect(Sx,rho) for rho in appro], 
             label=r"$\langle\Sigma_x \rangle$",color="#e41a1c", linestyle="--") 
    ax2.plot(tlist/args["tau"], [qt.expect(Sy,rho) for rho in appro], 
             label=r"$\langle\Sigma_y\rangle$",color="#377eb8", linestyle="--") 
    
    ax2.legend(edgecolor="black", framealpha=1, 
                        handlelength=1, borderpad=0.1, fontsize=12, loc=0, labelspacing=0.0) 

    ax2.set_xlabel(r"$s$")
    ax2.set_xticks([0, 1])
    ax2.set_xlim((0, 1))  # Hide x

    ###################

    color_list = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    color_list = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    lineS_list = ["-", "--", "-."]
    N_list = [3, 4, 5]

    for i in range(len(N_list)): 
        ax3.plot(tau_list, F_exp_g[i], color=color_list[i], linestyle=lineS_list[i], label=fr"$N={3+i}$")
    
    ax3.plot(tau_list, 5e6/tau_list**(3), color="black", linewidth=1)
    ax3.set_xscale("log") 
    ax3.set_yscale("log") 

    ax3.set_xlabel(r"$T\gamma$")
    ax3.set_ylabel(r"$D(\rho, \rho_\mathrm{ad})$")

    ax3.set_xlim((1, 5e4))
    ax3.set_xticks([1, 1e1, 1e2, 1e3, 1e4])

    ax3.set_ylim((1e-9, 1))
    #ax3.set_yticks([0.1, 10e-8])
    ax3.legend(edgecolor="black", framealpha=1, 
                         handlelength=1, borderpad=0.3, fontsize=10, loc=0, labelspacing=0.0) 




    ################# 

    for i in range(len(N_list)): 
        ax4.plot(tau_list, F_exp_gh[i], color=color_list[i] , linestyle=lineS_list[i], label=fr"$N={3+i}$")

    ax4.plot(tau_list, 1e4/tau_list**(2), color="black", linewidth=1)

    ax4.set_xscale("log") 
    ax4.set_yscale("log") 

    ax4.set_xlabel(r"$T\gamma$")
    ax4.set_ylabel(r"$D(\rho(T/2), \rho_\mathrm{ad}(T/2))$")

    ax4.set_xlim((1, 5e4))
    ax4.set_xticks([1, 1e1, 1e2, 1e3, 1e4])

    ax4.set_ylim((1e-6, 1))
    #ax4.set_yticks([0.1, 1e-8])
 
    ax4.legend(edgecolor="black", framealpha=1, 
                         handlelength=1, borderpad=0.3, fontsize=10, loc=0, labelspacing=0.0) 



    ############## 
    for i in range(len(N_list)): 
        ax5.plot(tau_list, F_exp_l[i],  color=color_list[i], linestyle=lineS_list[i], label=fr"$N={3+i}$")

    ax5.plot(tau_list, 5e5/tau_list**(3), color="black", linewidth=1)
    ax5.set_xscale("log") 
    ax5.set_yscale("log") 

    ax5.set_xlabel(r"$T\gamma$")
    ax5.set_ylabel(r"$D(\rho^1, \rho_\mathrm{ad}^1)$")


    ax5.set_xlim((1, 5e4))
    ax5.set_xticks([1, 1e1, 1e2, 1e3, 1e4])

    ax5.set_ylim((1e-9, 1))

    ax5.legend(edgecolor="black", framealpha=1, 
                         handlelength=1, borderpad=0.3, fontsize=10, loc=0, labelspacing=0.0) 
     

    plt.tight_layout()
    plt.savefig("Fig2.pdf", bbox_inches="tight") 
    #plt.show()
