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
from ManyPulses import *


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



def PlotFig3():

    C_Exact = np.load("Data/Coherence_exact.npy")
    C_Expon = np.load("Data/Coherence_expon.npy")
    C_First = np.load("Data/Coherence_first.npy")

    Se_Exact = np.load("Data/Density_exact.npy")
    Se_Expon = np.load("Data/Density_expon.npy")
    Se_First = np.load("Data/Density_first.npy")
 
    fig = plt.figure(figsize=(5.5, 5))

    # Define the main 2x2 grid

    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    gs_bottom_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.0)

    # Bottom-right top (C1)

    ax_C1 = fig.add_subplot(gs_bottom_right[0, 0], sharex=None)
    ax_C2 = fig.add_subplot(gs_bottom_right[1, 0], sharex=ax_C1)
    plt.setp(ax_C1.get_xticklabels(), visible=False)

    n_pulses = 10 
    N_list = np.arange(0, n_pulses+1) 
    #####
    rho_Exact,  rho_First = NPulses_dynamics(1000,n_pulses)
    ax_A.plot(N_list, [rho[3][3] for rho in rho_Exact], linestyle="-", marker="o", markeredgecolor="black", color="#e41a1c")
    ax_A.plot(N_list, [rho[2][2] for rho in rho_Exact], linestyle="-", marker="o", markeredgecolor="black", color="#377eb8")
    ax_A.plot(N_list, [rho[1][1] for rho in rho_Exact], linestyle="-", marker="o", markeredgecolor="black", color="#4daf4a")
    ax_A.plot(N_list, [rho[0][0] for rho in rho_Exact], linestyle="-", marker="o", markeredgecolor="black", color="#ff7f00")
    
    ax_A.plot(N_list, [rho[3][3] for rho in rho_First], linestyle=":", marker="^", markeredgecolor="black", color="#e41a1c")
    ax_A.plot(N_list, [rho[1][1] for rho in rho_First], linestyle=":", marker="^", markeredgecolor="black", color="#4daf4a")
    ax_A.plot(N_list, [rho[2][2] for rho in rho_First], linestyle=":", marker="^", markeredgecolor="black", color="#377eb8")
    ax_A.plot(N_list, [rho[0][0] for rho in rho_First], linestyle=":", marker="^", markeredgecolor="black", color="#ff7f00")


    ax_A.set_xlim((0, 10))
    ax_A.set_xlabel(r"$s$")

    ############
    ############
    ############

    rho_Exact, rho_First = NPulses_dynamics(10000,10)
    ax_B.plot(N_list, [rho[3][3] for rho in rho_Exact], linestyle="-", markeredgecolor="black",  marker="o", color="#e41a1c")
    ax_B.plot(N_list, [rho[2][2] for rho in rho_Exact], linestyle="-", markeredgecolor="black",  marker="o", color="#377eb8")
    ax_B.plot(N_list, [rho[1][1] for rho in rho_Exact], linestyle="-", markeredgecolor="black",  marker="o", color="#4daf4a")
    ax_B.plot(N_list, [rho[0][0] for rho in rho_Exact], linestyle="-", markeredgecolor="black",  marker="o", color="#ff7f00")

    ax_B.plot(N_list, [rho[3][3] for rho in rho_First], linestyle=":", markeredgecolor="black",  marker="^", label=r"$p_{00}$", color="#e41a1c")
    ax_B.plot(N_list, [rho[2][2] for rho in rho_First], linestyle=":", markeredgecolor="black",  marker="^", label=r"$p_{01}$", color="#377eb8")
    ax_B.plot(N_list, [rho[1][1] for rho in rho_First], linestyle=":", markeredgecolor="black",  marker="^", label=r"$p_{10}$", color="#4daf4a")
    ax_B.plot(N_list, [rho[0][0] for rho in rho_First], linestyle=":", markeredgecolor="black",  marker="^", label=r"$p_{11}$", color="#ff7f00")

    ax_B.set_xlim((0, 10))
    ax_B.set_xlabel(r"$s$")

    ax_B.legend(edgecolor="black", framealpha=1, 
               handlelength=1, borderpad=0.3, loc=0, labelspacing=0.0)

    ##########
    ##########
    ##########
    
    ax_C.plot(np.arange(len(Se_Exact[0])), np.array(Se_Exact[0])/4, linestyle="-",  marker="o",  markeredgecolor="black", color="black", label=r"$\mathrm{Exact}$")
    ax_C.plot(np.arange(len(Se_Exact[0])), np.array(Se_Expon[0])/4, linestyle="--", marker="s",  markeredgecolor="black", color="#e41a1c", label=r"$\mathrm{Exponential}$")
    ax_C.plot(np.arange(len(Se_Exact[0])), np.array(Se_First[0])/4, linestyle="-.", marker="^",  markeredgecolor="black", color="#377eb8", label=r"$\mathrm{First\; order}$")

    ax_C.plot(np.arange(len(Se_Exact[0])), np.array(Se_Exact[1])/4, linestyle="-",  marker="o",  markeredgecolor="black", color="black")
    ax_C.plot(np.arange(len(Se_Exact[0])), np.array(Se_Expon[1])/4, linestyle="--", marker="s",  markeredgecolor="black", color="#e41a1c")
    ax_C.plot(np.arange(len(Se_Exact[0])), np.array(Se_First[1])/4, linestyle="-.", marker="^",  markeredgecolor="black", color="#377eb8")

    ax_C.set_xlim((0, 20))
    ax_C.set_xlabel(r"$s$")

    ax_C.set_ylim(bottom=0)

    ax_C.legend(edgecolor="black", framealpha=1, 
               handlelength=1.5, borderpad=0.3, loc=0, labelspacing=0.0)

    ##########
    ##########
    ##########
    ax_C1.plot(np.arange(0.5, 0.5 + len(C_Exact[0])), C_Exact[0], linestyle="-",  marker="o", markeredgecolor="black", color="black")
    ax_C1.plot(np.arange(0.5, 0.5 + len(C_Exact[0])), C_Expon[0], linestyle="--", marker="s", markeredgecolor="black", color="#e41a1c")
    ax_C1.plot(np.arange(0.5, 0.5 + len(C_Exact[0])), C_First[0], linestyle="-.", marker="^", markeredgecolor="black", color="#377eb8")
    
    
    ax_C2.plot(np.arange(0.5, 0.5 + len(C_Exact[0])), C_Exact[1], linestyle="-",  marker="o", markeredgecolor="black", color="black", label=r"$\mathrm{Exact}$")
    ax_C2.plot(np.arange(0.5, 0.5 + len(C_Exact[0])), C_Expon[1], linestyle="--", marker="s", markeredgecolor="black", color="#e41a1c", label=r"$\mathrm{Exponential}$")
    ax_C2.plot(np.arange(0.5, 0.5 + len(C_Exact[0])), C_First[1], linestyle="-.", marker="^", markeredgecolor="black", color="#377eb8", label=r"$\mathrm{First\; order}$")
 
    ax_C2.set_xlim((0, 10))
    ax_C2.set_xlabel(r"$s$")

    ax_C1.set_ylim(bottom=0)
    ax_C2.set_ylim(bottom=0)

    ax_C2.legend(edgecolor="black", framealpha=1, 
               handlelength=1.5, borderpad=0.3, loc=0, labelspacing=0.0)

    plt.tight_layout()
    plt.savefig("Fig3.pdf", bbox_inches="tight")


def GetData():  

    args = GetParameters() 
    args["tau"] = 10000
    args["N"] = 5
    args["V"] = 3

    Np_p = 75
    Np_c = 15

    population_data  = Data_to_plot_Populations(Np_p, args.copy())
    coherences_data = Data_to_plot_Coherences(Np_c, args.copy())

    np.save("Data/population_data", np.array(population_data, dtype=object)) 
    np.save("Data/coherences_data", np.array(coherences_data, dtype=object)) 


