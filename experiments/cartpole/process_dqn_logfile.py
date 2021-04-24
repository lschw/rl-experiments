import os
import numpy as np
import tensorflow as tf
from tensorflow.core.util import event_pb2
import matplotlib.pyplot as plt

# Load and process the binary logfiles created by the hyperparameter search
# with cartpole_dqn.py




# Check if extracted parameters already exist
if os.path.isfile("params.dat") and os.path.isfile("returns.dat"):

    params = []
    returns = []

    # Load parameters and returns
    with open("params.dat") as fh:
        for line in fh:
            par = {}
            for p in line.strip("\n,").split(","):
                x = p.split("=")
                par[x[0]] = float(x[1])
            params.append(par)

    with open("returns.dat") as fh:
        for line in fh:
            returns.append([float(x) for x in line.strip().split(",")])

    # Create average over the 5 seeds
    params_avg = []
    returns_avg = []
    returns_std = []
    for i in range(int(len(params)/5)):
        returns_avg.append(np.mean(returns[i*5:(i+1)*5], axis=0))
        returns_std.append(np.std(returns[i*5:(i+1)*5], axis=0))
        params_avg.append(params[i*5])

    # Create selected parameter figure
    plt.figure(figsize=(6,4))
    i_selected = 2*(4 * 5) + 2*(5)
    for i,r in enumerate(returns[i_selected]):
        print(i, r)
    print(params[i_selected])
    plt.plot(range(100), returns[i_selected])
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.savefig("cartpole_return.png")

    # Create parameter study figure
    plt.figure(figsize=(10,8))
    for i in range(int(len(params_avg)/4)):
        ax = plt.subplot(3,3,i+1)
        ax.set_ylim(0,200)
        ax.set_xlim(0,100)
        if i < 3:
            ax.annotate("eps_decay={}".format(params_avg[i*4]["epsilon_decay"]),
                xy=(0.25,1.2), xycoords="axes fraction")

        if i > 5:
            ax.set_xlabel("Episodes")

        if i%3 == 0:
            ax.set_ylabel("Return")

            ax.annotate("gamma={}".format(params_avg[i*4]["gamma"]),
                xy=(-0.5,0.2), xycoords="axes fraction", rotation=90)

        ax.plot(range(100), returns_avg[i*4], label="t_u=1")
        ax.fill_between(range(100), returns_avg[i*4]-returns_std[i*4],
            returns_avg[i*4]+returns_std[i*4], alpha = 0.3)

        ax.plot(range(100), returns_avg[i*4+1], label="t_u=10")
        ax.fill_between(range(100), returns_avg[i*4+1]-returns_std[i*4+1],
            returns_avg[i*4+1]+returns_std[i*4+1], alpha = 0.3)

        ax.plot(range(100), returns_avg[i*4+2], label="t_u=20")
        ax.fill_between(range(100), returns_avg[i*4+2]-returns_std[i*4+2],
            returns_avg[i*4+2]+returns_std[i*4+2], alpha = 0.3)

        ax.plot(range(100), returns_avg[i*4+3], label="t_u=50")
        ax.fill_between(range(100), returns_avg[i*4+3]-returns_std[i*4+3],
            returns_avg[i*4+3]+returns_std[i*4+3], alpha = 0.3)
        ax.legend()
    plt.savefig("cartpole_dqn_learning.png")


# Extract parameters and returns from binary tensorflow summary logfile
else:
    fns = []
    for fn in os.listdir("./logs"):
        fns.append("./logs/"+fn)

    fns.sort()

    fns_full = []
    for fn in fns:
        for fn2 in os.listdir(fn):
            if fn2.startswith("events.out.tfevents"):
                fns_full.append(fn+"/"+fn2)

    fns = fns_full

    fh_p = open("params.dat", "w")
    fh_r = open("returns.dat", "w")

    for fn in fns:
        print(fn)
        par = {}
        ret = []
        dd = tf.data.TFRecordDataset(fn)
        for ds in dd:
            event = event_pb2.Event.FromString(ds.numpy())
            for value in event.summary.value:
                if value.tag == "hyperparameters":
                    for x in tf.make_ndarray(value.tensor):
                        par[x[0].decode("utf-8")] = float(x[1])
                if value.tag == "return":
                    print(event.step)
                    t = tf.make_ndarray(value.tensor).flatten()[0]
                    ret.append(t)


        fh_r.write(",".join([str(x) for x in ret]))
        fh_r.write("\n")
        fh_r.flush()

        for p in par:
            fh_p.write("{}={},".format(p,par[p]))
        fh_p.write("\n")
        fh_p.flush()
