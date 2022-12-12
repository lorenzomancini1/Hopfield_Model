import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit, fsolve
from scipy.optimize import differential_evolution
from scipy import stats
import warnings

def sigmoid(x, a, b, Offset):
    '''
    Sigmoid-like function to fit retrieval propabilities
    '''
    return 1.0 / (1.0 + np.exp(-1.0 * a * (x - b))) + Offset

def scaling(x, *params): 
    '''
    Scaling function to fit finite size effects
    '''
    return sum([p*(x**(i)) for i, p in enumerate(params)])

def plotsigmoid(x, y, N, popt):
    '''
    Function to plot the sigmoid fit
    '''
    
    fig = plt.figure(figsize = (6, 5))
    xx = np.linspace(min(x), max(x), 600)
    plt.plot(x, y, label = "simulation", marker = "o", color = "black", linestyle = "None", markerfacecolor = "None")
    plt.plot(xx, sigmoid(xx, *popt),
             label = "fit", color = "red", linewidth = 1.)
    plt.ylabel(r"$ \mathrm{P}( escape | \alpha )$", size = 12)
    plt.xlabel(r"$\alpha$", size = 12)
    plt.title("N = {}".format(N), size = 12)
    plt.legend()
    plt.grid(axis = "both", linestyle = "--", alpha = 0.5)
    plt.show()
    return

def plotscaling(x, y, yerr, popt, errorbar, λ = -1):
    '''
    Function to plot the finite size fit
    '''
    # get the reciproc of the sizes
    x = np.array(list(map(lambda n: 1/n, x)))

    fig = plt.figure(figsize = (7,5))

    xx = np.linspace(0, np.max(x), 600)

    # here y is the list of pcN reversed
    if errorbar:
        plt.errorbar(x, y, yerr = yerr, c = "black", marker = "o", markerfacecolor = "None", capsize = 3,
                linewidth = 1., linestyle = "None", label = r"experimental data")
    
    else:
        plt.plot(x, y, c = "black", marker = "o", markerfacecolor = "None",
                linewidth = 1., linestyle = "None", label = r"experimental data$")
    
    plt.plot(xx, scaling(xx, *popt), c = "blue",
             linewidth = 1., label = r"intercept $\alpha_c \sim$ {}".format(round(popt[0], 3)))
    plt.ylabel(r"$\alpha_c(N)$", size = 12)
    plt.xlabel(r"$\frac{1}{N}$", size = 12)
    plt.grid(axis = "both", linestyle = "--", alpha = 0.5)
    if λ > 0 : plt.title(r"\lambda = {}".format(λ), size = 12)
    plt.legend(fontsize = 12)
    plt.show()
    return

def compute_error(popt, vars):
    '''
    Function to propagate the errors of the solution x s. t. the fit is equal to 0.5
    '''
    d1 = np.log( 1 / (0.5 - popt[2]) - 1 ) / (popt[0]**2)
    d2 = 1
    d3 = (-1 / popt[0]) * 1 / ((0.5 - popt[2]) - (0.5 - popt[2])**2)
    derivates = np.array([d1, d2, d3])
    #print(derivates)
    error = np.sqrt( np.sum( ( (derivates**2)* vars ) ) )
    return error

def compute_critical_alpha(NN, rootdir = "./", datadir = "julia_data",
               plot_sig = False, plot_scal = False, λ = -1,
               errorbar = False, d = 2, skip = 0):

    αcN = [] # list to contain
    αcNerr = []

    for N in NN:
        if λ > 0: # if we have lambda > 0 we are working with the modern hopfield model
            folder = str(λ).replace(".", "")
            xData = np.loadtxt(rootdir+datadir+"/lambda_"+folder+"/N{}.txt".format(N), delimiter = "\t", skiprows = skip)[:, 0]
            yData = np.loadtxt(rootdir+datadir+"/lambda_"+folder+"/N{}.txt".format(N), delimiter = "\t", skiprows = skip)[:, 1]
            yErr = np.loadtxt(rootdir+datadir+"/lambda_"+folder+"/N{}.txt".format(N), delimiter = "\t", skiprows = skip)[:, 2]
        # otherwise we are working with the standard Hopfield
        else: 
            xData = np.loadtxt(rootdir+datadir+"/N{}.txt".format(N), delimiter = "\t", skiprows = skip)[:, 0]
            yData = np.loadtxt(rootdir+datadir+"/N{}.txt".format(N), delimiter = "\t", skiprows=skip)[:, 1]
            yErr = np.loadtxt(rootdir+datadir+"/N{}.txt".format(N), delimiter = "\t", skiprows=skip)[:, 2]

        yErr = np.array(list( map( lambda i: max(i, 10**(-3)), yErr ) ) )

        def sumOfSquaredError(parameterTuple):
            warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
            val = sigmoid(xData, *parameterTuple)
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

        popt, pcov = curve_fit(sigmoid, xData, yData, geneticParameters, maxfev = 10**5)#,
        #sigma = yErr, absolute_sigma=True)
        if plot_sig: plotsigmoid(xData, yData, N, popt)

        #find the x valure where the sigmoid is equal to 0.5
        sol = - np.log( 1/(0.5 - popt[2]) - 1)/popt[0] + popt[1]

        αcN.append(sol)
        αcNerr.append(compute_error(popt, np.diag(pcov)))
        
    x = list(map(lambda x: 1/x, NN))[::-1]
    y = αcN[::-1]
    popt, pcov = curve_fit(scaling, x, y, p0 = [1]*(d+1), sigma = αcNerr)
    if plot_scal: plotscaling(NN, αcN, αcNerr, popt, errorbar, λ = λ)
    
    return αcN, αcNerr, popt[0], np.sqrt(pcov[0,0]) 