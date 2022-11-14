import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit, fsolve
from scipy.optimize import differential_evolution
from scipy import stats
import warnings

def func(x, a, b, Offset):
    return 1.0 / (1.0 + np.exp(-1.0 * a * (x - b))) + Offset

def reg_func(x, *params): 
    return sum([p*(x**(i)) for i, p in enumerate(params)])

def plotreg(NN, pcN, pcNerr, popt, errorbar):
    N_reciproc = np.array(list(map(lambda n: 1/n, NN)))
    fig = plt.figure(figsize = (7,5))
    x = np.linspace(0, np.max(N_reciproc), 600)
    #plt.plot(N_reciproc, pcN, c = "black", marker = "o", markerfacecolor = "None",
    #         linewidth = 1., linestyle = "None", label = r"experimental $p_c(N, \alpha = {})$".format(α))
    if errorbar:
        plt.errorbar(N_reciproc, pcN, yerr = pcNerr, c = "black", marker = "o", markerfacecolor = "None", capsize = 3,
                linewidth = 1., linestyle = "None", label = r"experimental $p_c(N)$")
    
    else:
        plt.plot(N_reciproc, pcN, c = "black", marker = "o", markerfacecolor = "None",
                linewidth = 1., linestyle = "None", label = r"experimental $p_c(N)$")
    
    plt.plot(x, sum([p*(x**(i)) for i, p in enumerate(popt)]), c = "blue",
             linewidth = 1., label = r"intercept $\sim$ {}".format(round(popt[0], 3)))
    #plt.xlabel(r"1/$\sqrt{N}$", size = 12)
    plt.ylabel(r"$p_c(N)$", size = 12)
    plt.xlabel(r"$\frac{1}{N}$", size = 12)
    plt.grid(axis = "both", linestyle = "--", alpha = 0.5)
    plt.title("Finite size scaling", size = 12)
    plt.legend(fontsize = 12)
    plt.show()
    return

def plotfit(x, N, y, popt):
    fig = plt.figure(figsize = (6, 5))
    xx = np.linspace(min(x), max(x), 600)
    plt.plot(x, y, label = "simulation", marker = "o", color = "black", linestyle = "None", markerfacecolor = "None")
    plt.plot(xx, func(xx, *popt),
             label = "fit", color = "red", linewidth = 1.)
    plt.ylabel("Reconstruction prob.", size = 12)
    plt.xlabel("p", size = 12)
    plt.title("N = {}".format(N), size = 12)
    plt.legend()
    plt.grid(axis = "both", linestyle = "--", alpha = 0.5)
    plt.show()
    return

def compute_error_new(popt, vars):
    d1 = np.log( 1 / (0.5 - popt[2]) - 1 ) / (popt[0]**2)
    d2 = 1
    d3 = (-1 / popt[0]) * 1 / ((0.5 - popt[2]) - (0.5 - popt[2])**2)
    derivates = np.array([d1, d2, d3])
    #print(derivates)
    error = np.sqrt( np.sum( ( (derivates**2)* vars ) ) )
    return error

def compute_αc(NN, rootdir = "./", datadir = "julia_data",
               plot_fit = False, plot_reg = False,
               errorbar = False, d = 2, skip = 0):

    αcN = []
    αcNerr = []
    xData = np.loadtxt(rootdir+datadir+"/N{}.txt".format(NN[0]), delimiter = "\t", skiprows = skip)[:, 0]

    for N in NN:
        yData = np.loadtxt(rootdir+datadir+"/N{}.txt".format(N), delimiter = "\t", skiprows = skip)[:, 1]

        yErr = np.loadtxt(rootdir+datadir+"/N{}.txt".format(N), delimiter = "\t", skiprows = skip)[:, 2]

        yErr = np.array(list( map( lambda i: max(i, 10**(-3)), yErr ) ) )

        def sumOfSquaredError(parameterTuple):
            warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
            val = func(xData, *parameterTuple)
            return np.sum((yData - val) ** 2.0)

        maxX = max(xData)
        minX = min(xData)
        maxY = max(yData)
        minY = min(yData)

        parameterBounds = []
        parameterBounds.append([minX, maxX]) # search bounds for a
        parameterBounds.append([minX, maxX]) # search bounds for b
        parameterBounds.append([minY, maxY]) # search bounds for Offset

        # "seed" the np random number generator for repeatable results
        result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
        geneticParameters = result.x

        popt, pcov = curve_fit(func, xData, yData, geneticParameters, maxfev = 10**5)

        if plot_fit: plotfit(xData, N, yData, popt)
        #x = np.linspace(-10, 10, 300)
        #plt.plot(x, func(x, *popt))
        #sol = optimize.root_scalar(lambda x: func(x, *popt) - 0.5, bracket=[xi, xf], method='brentq', x0 = x_guess)

        sol = - np.log( 1/(0.5 - popt[2]) - 1)/popt[0] + popt[1]
        #print(sol.root)
        #print(sol0)
       #print(pcov)
        αcN.append(sol)
        αcNerr.append(compute_error_new(popt, np.diag(pcov)))
    
    x = list(map(lambda x: 1/x, NN))[::-1]
    y = αcN[::-1]
    popt, pcov = curve_fit(reg_func, x, y, p0 = [1]*(d+1), sigma = αcNerr)
    if plot_reg: plotreg(NN, αcN, αcNerr, popt, errorbar)

    return αcN, αcNerr, popt[0], np.sqrt(pcov[0,0])