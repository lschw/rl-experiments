import sys
sys.path.insert(0, "../../")
import numpy as np
import gym
import algorithms as alg

env = gym.make("CartPole-v0")

bd = alg.encoding.BoxDiscretization(
    env.observation_space,
    N_buckets=[10,10,10,10],
    limits=[
        [env.observation_space.low[0],env.observation_space.high[0]],
        [-2,2],
        [env.observation_space.low[2],env.observation_space.high[2]],
        [-2,2]
    ]
)
env = gym.wrappers.TransformObservation(env, bd.encode)
sagg = alg.encoding.StateAggregation(bd.N, 1, env.action_space.n)

print("Semi-gradient Q-Learning")
alg.utils.random_seed(env, 1)
w = sagg.generate_weights()
w,history_sg_qlearning = alg.semi_gradient_qlearning(
    env, sagg.q, sagg.q_deriv, w, alpha=0.1, gamma=1, epsilon=0.8,
    N_episodes=3000, epsilon_decay=alg.utils.decay_sigmoid)

alg.utils.plot_learning_curves(
    [history_sg_qlearning],
    ["Semi-gradient Q-Learning"],
    "cartpole_sg_learning.pdf"
)

for i in range(10):
    done = False
    state = env.reset()
    steps = 1
    ret = 0
    while not done:
        Q_s = [sagg.q(state, a, w) for a in range(env.action_space.n)]
        action = alg.utils.select_action_greedy(Q_s)
        state, reward, done, info = env.step(action)
        env.render()
        steps += 1
        ret += reward
    print("Episode {}: Steps {}, Return {}".format(i, steps, ret))
