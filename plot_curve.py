import matplotlib.pyplot as plt
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')

def plot_curve(x, y_list, title, curve_name, label_name):
    plt.title(title)
    x_name, y_name = label_name
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    for y in y_list:
        plt.plot(x, y)
    plt.legend(curve_name, loc="best")
    plt.show()

time = joblib.load("dqn_result/dqn_t.dat")
mean = joblib.load("dqn_result/dqn_mean.dat")
best = joblib.load("dqn_result/dqn_best.dat")

plot_curve(time, [mean, best], "DQN Pacman Learning Curve",["dqn_mean", "dqn_best"], ("Iteration", "reward"))
