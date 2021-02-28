import sys
sys.path.insert(0, "../../")
import numpy as np
from collections import defaultdict
import gym
from environments import gym_randomwalk
import algorithms as alg
import numpy as np
import matplotlib.pyplot as plt

def plot_value_function(vs, labels):
    fig = plt.figure(figsize=(6,4))
    ax = plt.subplot(1, 1, 1)
    ax.set_ylabel("Value")
    ax.set_xlabel("State")
    for i,v in enumerate(vs):
        ax.plot(range(1,len(v)-1), v[1:-1], label=labels[i])
    plt.legend(frameon=False)
    plt.savefig("randomwalk.pdf")


env = gym.make("RandomWalk-v0", N=1000, step_range=100, terminal_rewards=(-1,1))

pi = np.ones((env.observation_space.n,env.action_space.n))/env.action_space.n

vs=[]
labels=[]

print("DP policy evaluation")
alg.utils.random_seed(env, 1)
labels.append("Policy evaluation")
vs.append(
    alg.dp_policy_evaluation(env, pi, gamma=1, tol=1e-5, iter_max=300)
)

print("TD(0)")
labels.append("TD(0)")
alg.utils.random_seed(env, 1)
v = alg.td0(
    env, pi, alpha=0.1, gamma=1, N_episodes=50000, ep_max_length=1000,
    alpha_decay=alg.utils.decay_linear)
vs.append([v[i] for i in range(len(v))])

print("Gradient MC evaluation")
labels.append("Gradient MC evaluation")
sagg = alg.encoding.StateAggregation(N_states=1000, group_size=100)
w = sagg.generate_weights()
w = alg.gradient_mc_evaluation(
    env, pi, sagg.v, sagg.v_deriv, w, gamma=1, alpha=0.01,
    N_episodes=20000, ep_max_length=1000, alpha_decay=alg.utils.decay_linear)
vs.append([sagg.v(s,w) for s in range(env.nS)])
print(w)

print("Semi-gradient TD(0)")
labels.append("Semi-gradient TD(0)")
sagg = alg.encoding.StateAggregation(N_states=1000, group_size=100)
w = sagg.generate_weights()
w = alg.semi_gradient_td0(
    env, pi, sagg.v, sagg.v_deriv, w, gamma=1, alpha=0.1,
    N_episodes=10000, ep_max_length=1000, alpha_decay=alg.utils.decay_linear)
vs.append([sagg.v(s,w) for s in range(env.nS)])
print(w)

plot_value_function(vs, labels)

