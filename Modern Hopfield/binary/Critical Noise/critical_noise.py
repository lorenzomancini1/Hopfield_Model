import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit, fsolve
from scipy import stats
from sklearn.linear_model import LinearRegression

def reg_func(x, q, a, b): return q + a*x**(-1) + b*x**(-2) # function for finite size scaling, i.e. fit the critical probabilities
    
def plotfit(pp, N, P_N, popt):
    fig = plt.figure(figsize = (6, 5))
    x = np.linspace(min(pp), max(pp), 600)
    plt.plot(pp, P_N, label = "simulation", marker = "o", color = "black", linestyle = "None", markerfacecolor = "None")
    plt.plot(x, np.exp(popt[0] + (popt[1] * x) + (popt[2] * x**2) + (popt[3] * x**3) + (popt[4] * x**4) + (popt[5] * x**5) + (popt[6] * x**6)),
             label = "fit", color = "red", linewidth = 1.)
    plt.ylabel("Reconstruction prob.", size = 12)
    plt.xlabel("p", size = 12)
    plt.title("N = {}".format(N), size = 12)
    plt.legend()
    plt.grid(axis = "both", linestyle = "--", alpha = 0.5)
    plt.show()
    return

def plotreg(N_reciproc, pcN, popt, α):
    fig = plt.figure(figsize = (7,5))
    x = np.linspace(0, np.max(N_reciproc), 600)
    plt.plot(N_reciproc, pcN, c = "black", marker = "o", markerfacecolor = "None",
             linewidth = 1., linestyle = "None", label = r"experimental $p_c(N, \alpha = {})$".format(α))
    plt.plot(x, popt[0] + popt[1]*x + popt[2] * x**2, c = "blue",
             linewidth = 1., label = r"intercept $p_c(\alpha = {}) \sim$ {}".format(α, round(popt[0], 3)))
    #plt.xlabel(r"1/$\sqrt{N}$", size = 12)
    plt.ylabel(r"$p_c(N, \alpha = {})$".format(α), size = 12)
    plt.xlabel(r"$\frac{1}{N}$", size = 12)
    plt.grid(axis = "both", linestyle = "--", alpha = 0.5)
    plt.title("Finite size scaling", size = 12)
    plt.legend(fontsize = 12)
    plt.show()
    return

def P_rec(p, q, a, b, c, d, e, f): 
    return np.exp(q + (a * p) + (b * p**2) + (c * p**3) + (d * p**4) + (e * p**5) + (f * p**6)) # function to fit the reconstruction probability
 

def compute_pc(α, NN, rootdir = "./", datadir = "workstation_data/reconstruction_prob",
               plot_fit = False, plot_reg = False,
               thr = 0.5,
               x_guess = 0.3,
               xi = 0.15,
               xf = 0.45):
    
    pcN = []
    folder = str(α).replace(".", "")
    pp = np.loadtxt(rootdir+datadir+"/alpha_"+folder+"/probsN{}.txt".format(NN[0]), delimiter = "\t")[:, 0]
        
    for N in NN:
        
        P_N = np.loadtxt(rootdir+datadir+"/alpha_"+folder+"/probsN{}.txt".format(N), delimiter = "\t")[:, 1] # get Probs data
        e_N = np.loadtxt(rootdir+datadir+"/alpha_"+folder+"/probsN{}.txt".format(N), delimiter = "\t")[:, 2] # get Probs errors
        popt, pcov = curve_fit(P_rec, pp, P_N, maxfev = 5000) # fit the Probs data with the exponential defined in P_rec
        
        if plot_fit: plotfit(pp, N, P_N, popt) # show the fit                   
        
        # find the value of p that makes P_rec = 0.5
        sol = optimize.root_scalar(lambda p: np.exp(popt[0] + popt[1] * p + popt[2] * p**2 + popt[3] * p**3 + 
                                                    popt[4] * p**4 + popt[5] * p**5 + popt[6] * p**6) - thr,
        x0 = x_guess, bracket = [xi, xf], method='brentq') # find the value for p that makes the fit be equal to 0.5
    
        pcN.append(sol.root) # store the result in the array
    
    pcN = pcN#[::-1] # this line can be deleted
    N_reciproc = np.array(list(map(lambda n: 1/n, NN)))#[::-1] # array of 1/N
    popt, pcov = curve_fit(reg_func, NN, pcN) # fit the results with reg_func
    
    if plot_reg: plotreg(N_reciproc, pcN, popt, α) # show the result
        
        
    return pcN, popt[0]

if __name__ == "__main__":
    NN = [50, 100, 150, 200, 1000]
    α = 0.1
    compute_pc(α, NN, plot_fit= False, plot_reg = False)


