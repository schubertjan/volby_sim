import numpy as np
import statsmodels.api as sm
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

def logit(x):
    #make sure it works even if not a np array supplied
    x_ = [1e-4 if i==0 else i for i in x]    
    #make an array
    x_ = np.array(x_)
    l = np.log(x_ / (1-x_))
    return l

def sigmoid(x):
    s = 1 / (1 + np.exp(1) ** -x)
    return s

def extract_diff(seats, votes):
    d = seats - votes

    # calculate when (on average) parties benefit
    y = d.flatten()
    X = votes.flatten()
    # select only cases which pass the threshold
    y = y[X >= 0.05]
    X = X[X >= 0.05]

    #fit a model
    X = sm.add_constant(X)
    results = sm.OLS(y, X).fit()
    a_, b_ = results.params
    y_plus = np.round(-1 * a_/b_, 2)

    return {"d": y, "v": X[:,1], "d0": y_plus, "a": a_, "b": b_, "m": results}

def plot_diff(d,v, d0, tItle):
    fig, ax = plt.subplots(1,1, figsize = (10,5))
    ax.scatter(v, d, alpha = 0.05)
    ax.arrow(d0, 0.025, dx = 0, dy = -0.02, color = "black")
    ax.text(d0, 0.03, s = "Získává více mandátů: {}".format(d0), horizontalalignment = "center")
    ax.set_title(tItle)
    ax.set_ylabel("Rozdíl: mandáty - hlasy (p.b.)")
    ax.set_xlabel("% hlasů")
    ax.plot(np.linspace(-0.05,0.5), np.linspace(0,0), color = "black", linestyle = ":")
    ax.plot(np.linspace(0.05,0.05), np.linspace(-0.05,0.125), color = "black", linestyle = ":")
    ax.add_patch(Rectangle((-0.05, -0.05), 0.1, 0.175, facecolor = 'grey', alpha = 0.1))
    ax.add_patch(Rectangle((0.05, -0.05), 0.45, 0.05, facecolor = 'red', alpha = 0.1))
    ax.add_patch(Rectangle((0.05, 0), 0.45, 0.125, facecolor = 'green', alpha = 0.1))
    ax.text(0, 0.115, s = "5% hranice", horizontalalignment = "center")
    ax.text(0.15, 0.115, s = "Získává mandáty", horizontalalignment = "center")
    ax.text(0.45, -0.045, s = "Ztrácí mandáty", horizontalalignment = "center")
