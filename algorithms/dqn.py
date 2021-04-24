import numpy as np
from collections import deque
import tensorflow as tf
import random

def dqn(env, model, gamma, epsilon, epsilon_decay, epsilon_min, episodes,
    buffer_size, batch_size, target_update_freq, checkpoint_freq,
    checkpoint_path):
    """Implementation of the DQN algorithm to learn an estimate of the optimal
    value function using experience replay and a separate target network

    Based on
    * https://arxiv.org/abs/1312.5602
    * https://www.nature.com/articles/nature14236

    Args:
        env: Environment
        model: Keras model to optimize
        gamma: Discount factor
        epsilon: Parameter for epsilon-greedy policy
        epsilon_decay: Epsilon decay factor
        epsilon_min: Minimum value for epsilon
        episodes: Run this many episodes
        buffer_size: Experience replay buffer size
        batch_size: Batch size of model update
        target_update_freq: Target model update frequency
        checkpoint_freq: Save model parameter frequency
        checkpoint_path: Filepath for saving model parameters
    """
    steps = 0
    memory = deque(maxlen=buffer_size)
    target_model = tf.keras.models.clone_model(model)
    for i_episode in range(episodes):
        tf.summary.scalar("epsilon", epsilon, step=i_episode)
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            print("\r> DQN: Episode {}/{}, Step {}, Return {}".format(
                i_episode+1, episodes, steps, episode_return), end="")

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(np.array([state]))[0])

            # Take action and obtain observation and reward
            state_new, reward, done, info = env.step(action)
            episode_return += reward

            # Save experience
            memory.append((state,action,reward,state_new,done))

            # Experience replay
            if len(memory) > batch_size:
                experience_sample = random.sample(memory, batch_size)
                x = np.array([e[0] for e in experience_sample])

                # Construct target
                y = model.predict(x)
                x2 = np.array([e[3] for e in experience_sample])
                Q2 = gamma*np.max(target_model.predict(x2), axis=1)
                for i,(s,a,r,s2,d) in enumerate(experience_sample):
                    y[i][a] = r
                    if not d:
                        y[i][a] += Q2[i]

                # Update
                model.fit(x, y, batch_size=batch_size, epochs=1, verbose=0)

                # Save weight histogram
                for layer in model.layers:
                    for weight in layer.weights:
                        weight_name = weight.name.replace(':', '_')
                        tf.summary.histogram(weight_name, weight, step=steps)

            # Update of target model
            if steps % target_update_freq == 0:
                target_model.set_weights(model.get_weights())

            # Save model checkpoint
            if steps % checkpoint_freq == 0:
                model.save_weights("{}/weights-{:08d}-{:08d}".format(
                    checkpoint_path, i_episode, steps))

            state = state_new
            steps += 1

        # Epsilon decay
        epsilon *= epsilon_decay
        epsilon = max(epsilon_min, epsilon)

        tf.summary.scalar("return", episode_return, step=i_episode)
        tf.summary.flush()

        # Save final weights
        if steps-1 % checkpoint_freq != 0:
            model.save_weights("{}/weights-{:08d}-{:08d}".format(
                checkpoint_path, i_episode, steps-1))
    print()
