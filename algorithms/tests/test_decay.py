import sys
sys.path.insert(0, "../../")
import matplotlib.pyplot as plt
import algorithms as alg


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
