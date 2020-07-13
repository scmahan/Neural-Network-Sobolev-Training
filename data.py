import matplotlib.pyplot as plt
import numpy as np
    
def Softsign():
    def f(x):
        val = x/(1+np.abs(x))
        return val
    return f

def SSderiv():
    def f(x):
        val = 1/((1+np.abs(x))**2)
        return val
    return f

def SSderiv2():
    def f(x):
        val = -2*np.sign(x)/((1+np.abs(x))**3)
        return val
    return f

def genTrainData_SS(num_samples=1024):
    fn = Softsign()
    fnd = SSderiv()
    fndd = SSderiv2()
    samples = []
    for n in range(num_samples):
        x = np.array([np.random.uniform(-5,5)])
        y = fn(x)
        dy = fnd(x)
        d2y = fndd(x)
        s = (x, y, dy, d2y)
        samples.append(s)
    return samples

def plotSS():
    fn = Softsign()
    x = np.arange(-5, 5, 0.25)
    y = fn(x)
    plt.plot(x,y)
    
def plotSSderiv():
    fn = SSderiv()
    x = np.arange(-5, 5, 0.25)
    y = fn(x)
    plt.plot(x,y)
    
    