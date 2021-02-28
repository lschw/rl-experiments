import sys
sys.path.insert(0, "../../")
import matplotlib.pyplot as plt
import algorithms as alg
import gym
import numpy as np

box = gym.spaces.box.Box(
    low=np.array([-1.2, -0.07], dtype=np.float32),
    high=np.array([0.6, 0.07], dtype=np.float32),
    dtype=np.float32)

sd = alg.encoding.BoxDiscretization(box,
    N_buckets=[3,4],
    limits=[[-1.2,0.6],[-0.07,0.07]]
)

encoding = np.zeros([10,10])
data1 = np.linspace(sd.limits[0][0], sd.limits[0][1], encoding.shape[0])
data2 = np.linspace(sd.limits[1][0], sd.limits[1][1], encoding.shape[1])
for i,ival in enumerate(data1):
    for j,jval in enumerate(data2):
        encoding[i,j] = sd.encode((ival,jval))

plt.figure()
fig, ax = plt.subplots()
im = plt.imshow(encoding)
ax.set_yticks(np.arange(encoding.shape[0]))
ax.set_xticks(np.arange(encoding.shape[1]))
ax.set_xticklabels(["{:.2f}".format(x) for x in data1])
ax.set_yticklabels(["{:.2f}".format(x) for x in data2])
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range( encoding.shape[0]):
    for j in range( encoding.shape[1]):
        text = ax.text(j, i, "{:.0f}".format(encoding[i, j]),
            ha="center", va="center", color="w")

plt.savefig("test_box_discretization.pdf")




















fig = plt.figure(figsize=(6,4))
plt.xlabel("Episodes")
plt.ylabel("Value")
eps = 0.5
N = 1000
eps_non = [alg.utils.decay_none(eps, i, N) for i in range(N)]
eps_lin = [alg.utils.decay_linear(eps, i, N) for i in range(N)]
eps_exp = [alg.utils.decay_exponential(eps, i, N) for i in range(N)]
eps_sig = [alg.utils.decay_sigmoid(eps, i, N) for i in range(N)]
plt.plot(range(N), eps_non, label="none")
plt.plot(range(N), eps_lin, label="linear")
plt.plot(range(N), eps_exp, label="exponential")
plt.plot(range(N), eps_sig, label="sigmoid")
plt.legend(loc="lower left", frameon=False)
plt.savefig("test_decay.pdf", dpi=200)
