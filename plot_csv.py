import matplotlib.legend
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d

"""
Use this class to create plots from csv files from tensorboard

"""
parser = argparse.ArgumentParser(description='Setup your Environment.')
parser.add_argument("-f", "--folder", help="Enter folder, where plots are", type=str)
parser.add_argument("-s", "--sigma", help="Enter smoothing factor sigma. Default 1 not smooth. 2 smoother", default=1, type=int)
args = parser.parse_args()

plt.figure(figsize=(19,10))
folder = args.folder
if os.path.isdir(folder):
    files = os.listdir(folder)
    for file in files:
        if "Value Loss" in file:
            value_loss = pd.read_csv(folder + "\\" + file, index_col=1)
            vals = [i[1] for i in value_loss.values]
            vals = gaussian_filter1d(vals, sigma=args.sigma)
            index = value_loss.index
            value_loss = pd.DataFrame(data=vals, index=index)
            plt.plot(value_loss, label="Value_Loss")
        elif "Entropy Loss" in file:
            entropy_loss = pd.read_csv(folder + "\\" + file, index_col=1)
            vals = [i[1] for i in entropy_loss.values]
            vals = gaussian_filter1d(vals, sigma=args.sigma)
            index = entropy_loss.index
            entropy_loss = pd.DataFrame(data=vals, index=index)
            plt.plot(entropy_loss, label="Entropy_Loss")
        elif "Policy Loss" in file:
            policy_loss = pd.read_csv(folder + "\\" + file, index_col=1)
            vals = [i[1] for i in policy_loss.values]
            vals = gaussian_filter1d(vals, sigma=args.sigma)
            index = policy_loss.index
            policy_loss = pd.DataFrame(data=vals, index=index)
            plt.plot(policy_loss, label="Policy_Loss")
        else:
            episode_loss = pd.read_csv(folder + "\\" + file, index_col=1)
            vals = [i[1] for i in episode_loss.values]
            vals = gaussian_filter1d(vals, sigma=args.sigma)
            index = episode_loss.index
            episode_loss = pd.DataFrame(data=vals, index=index)
            plt.plot(episode_loss, label="Episode_Loss")
    plt.legend(loc="upper left")
    axes = plt.gca()
    axes.set_ylim([-1, 8])
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.show()
