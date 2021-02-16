import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curves(histories, labels, filename, avg=10, ylim_ret=None,
        ylim_epl=None):
    data = {}
    for i_h,history in enumerate(histories):
        returns = [0]*len(history)
        ep_length = [0]*len(history)
        for i in range(len(history)):
            ep_length[i] = len(history[i])
            for s,a,r in history[i]:
                returns[i] += r

        # Create moving average
        returns = np.convolve(returns, np.ones(avg), "valid") / avg
        ep_length = np.convolve(ep_length, np.ones(avg), "valid") / avg
        data[labels[i_h]] = {
            "returns": returns, "ep_length": ep_length
        }


    fig = plt.figure(figsize=(6,4))
    ax = plt.subplot(2, 1, 1)
    ax.set_ylabel("Return")
    if ylim_ret != None:
        ax.set_ylim(*ylim_ret)
    for label in data:
        ax.plot(range(len(data[label]["returns"])), data[label]["returns"],
            label=label)
    plt.legend(frameon=False)

    ax = plt.subplot(2, 1, 2)
    ax.set_ylabel("Episode length")
    ax.set_xlabel("Episode")
    if ylim_epl != None:
        ax.set_ylim(*ylim_epl)
    for label in data:
        ax.plot(range(len(data[label]["ep_length"])), data[label]["ep_length"],
            label=label)
    plt.legend(frameon=False)

    plt.savefig(filename, dpi=200)
