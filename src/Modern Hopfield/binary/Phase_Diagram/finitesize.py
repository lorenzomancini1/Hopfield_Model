import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit, fsolve
from scipy import stats

def reg_func(x, *params): 
    return sum([p*(x**(i)) for i, p in enumerate(params)])


def P_esc(x, *params): 
    return np.exp(sum([p*(x**i) for i, p in enumerate(params)]))

def plotfit(xx, y, N, popt):
    fig = plt.figure(figsize = (6, 5))
    x = np.linspace(min(xx), max(xx), 600)
    plt.plot(xx, y, label = "simulation", marker = "o", color = "black", linestyle = "None", markerfacecolor = "None")
    plt.plot(x, np.exp(sum([p*(x**i) for i, p in enumerate(popt)])),
             label = "fit", color = "red", linewidth = 1.)
    plt.ylabel("Prob.", size = 12)
    plt.xlabel("α", size = 12)
    plt.title("N = {}".format(N), size = 12)
    plt.legend()
    plt.grid(axis = "both", linestyle = "--", alpha = 0.5)
    plt.show()
    return

def plotreg(NN, y, yerr, popt, λ, errorbar):
    N_reciproc = np.array(list(map(lambda n: 1/n, NN)))
    fig = plt.figure(figsize = (7,5))
    x = np.linspace(0, np.max(N_reciproc), 600)
    #plt.plot(N_reciproc, y, c = "black", marker = "o", markerfacecolor = "None",
    #         linewidth = 1., linestyle = "None", label = r"experimental $p_c(N, \alpha = {})$".format(λ))
    if errorbar:
        plt.errorbar(N_reciproc, y, yerr = yerr, c = "black", marker = "o", markerfacecolor = "None",
                linewidth = 1., linestyle = "None", label = r"experimental $α_c(N, \lambda = {})$".format(λ))
    
    else:
        plt.plot(N_reciproc, y, c = "black", marker = "o", markerfacecolor = "None",
                linewidth = 1., linestyle = "None", label = r"experimental $α_c(N, \lambda = {})$".format(λ))
    
    plt.plot(x, sum([p*(x**(i)) for i, p in enumerate(popt)]), c = "blue",
             linewidth = 1., label = r"intercept $α_c(\lambda = {}) \sim$ {}".format(λ, round(popt[0], 3)))
    #plt.xlabel(r"1/$\sqrt{N}$", size = 12)
    plt.ylabel(r"$α_c(N, \lambda = {})$".format(λ), size = 12)
    plt.xlabel(r"$\frac{1}{N}$", size = 12)
    plt.grid(axis = "both", linestyle = "--", alpha = 0.5)
    plt.title("Finite size scaling", size = 12)
    plt.legend(fontsize = 12)
    plt.show()
    return

def compute_error(popt0, stds, x_guess, thr, nsamples = 10**4):
    pc_distribution = np.zeros(nsamples)
    #print(stds)
    l = len(popt0)
    for s in range(nsamples):
        popt = np.zeros(l)
        for i in range(l):
            popt[i] = np.random.normal(loc = popt0[i], scale = stds[i])
        root = fsolve(lambda x: np.exp(sum([p*(x**i) for i, p in enumerate(popt)])) - thr, x0 = x_guess, maxfev = 5000)
        pc_distribution[s] = root[0]
    return np.std(pc_distribution)

def compute_αc(λ, NN, rootdir = "./", datadir = "julia_data",
               plot_fit = False, plot_reg = False,
               thr = 0.5,
               x_guess = 0.3, dret = 6, dfss = 2,
               errorbar = False):
    
    pcN = []
    pcNerr = []
    folder = str(λ).replace(".", "")
    αα = np.loadtxt(rootdir+datadir+"/lambda_"+folder+"/N{}.txt".format(NN[0]), delimiter = "\t")[:, 0]
        
    for N in NN:
        P_N = np.loadtxt(rootdir+datadir+"/lambda_"+folder+"/N{}.txt".format(N), delimiter = "\t")[:, 1] # get Probs data
        #e_N = np.loadtxt(rootdir+datadir+"/alpha_"+folder+"/N{}.txt".format(N), delimiter = "\t")[:, 2] # get Probs errors
        popt, pcov = curve_fit(P_esc, αα, P_N, p0 = [1]*(dret+1), maxfev = 2*10**4) # fit the Probs data with the exponential defined in P_rec
        
        if plot_fit: plotfit(αα, P_N, N, popt) # show the fit                   
        
        # find the value of p that makes P_rec = 0.5
        #root = fsolve(lambda x: np.exp(sum([p*(x**i) for i, p in enumerate(popt)])) - thr, x_guess)
        xi = x_guess * (1.2)
        xf = x_guess * (0.8)
        sol = optimize.root_scalar(lambda x: np.exp(sum([p*(x**i) for i, p in enumerate(popt)])) - thr, bracket=[xi, xf], method='brentq', x0 = x_guess)
        pcN.append(sol.root) # store the result in the array
        #pcNerr.append(compute_error(popt, np.sqrt(np.diag(pcov)), x_guess, thr, nsamples = 10**4))
        #pcNerr.append(compute_error_new(root[0], popt, np.diag(pcov)))
    
    ##[::-1] # array of 1/N
    x = list(map(lambda x: 1/x, NN))[::-1]
    y = pcN[::-1]
    popt, pcov = curve_fit(reg_func, x, y, p0 = [1]*(dfss+1))#, sigma = pcNerr) # fit the results with reg_func
    
    if plot_reg: plotreg(NN, pcN, pcNerr, popt, λ, errorbar) # show the result
    
    return pcN, pcNerr, popt[0], np.sqrt(pcov[0,0])