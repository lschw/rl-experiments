import sys
sys.path.insert(0, "../../")
import numpy as np
import gym
import algorithms as alg

env = gym.make("MountainCar-v0")

bd = alg.encoding.BoxDiscretization(env.observation_space,
    N_buckets=[10,10],
    limits=[
        [env.observation_space.low[0],env.observation_space.high[0]],
        [env.observation_space.low[1],env.observation_space.high[1]]
    ]
)
env = gym.wrappers.TransformObservation(env, bd.encode)

print("Q-Learning")
alg.utils.random_seed(env, 1)
Q,history_qlearning = alg.qlearning(
    env, alpha=0.1, gamma=1, epsilon=0.1, N_episodes=2000)

alg.utils.plot_learning_curves(
    [history_qlearning],
    ["Q-Learning"],
    "mountaincar_td_learning.pdf"
)

for i in range(10):
    done = False
    state = env.reset()
    steps = 1
    ret = 0
    while not done:
        action = alg.utils.select_action_greedy(Q[state])
        state, reward, done, info = env.step(action)
        env.render()
        steps += 1
        ret += reward
    print("Episode {}: Steps {}, Return {}".format(i, steps, ret))
