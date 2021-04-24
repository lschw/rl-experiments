import sys
sys.path.insert(0, "../../")
import os
import random
import numpy as np
from datetime import datetime
import tensorflow as tf
import gym
from algorithms import dqn

def set_seed(env, seed=1):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


def create_model(env, layers, lr):
    state_dim = env.observation_space.shape[0]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
        layers[0], input_dim=state_dim, activation="relu"))
    for i in range(1,len(layers)):
        model.add(tf.keras.layers.Dense(layers[i], activation="relu"))
    model.add(tf.keras.layers.Dense(env.action_space.n))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=lr))
    model.summary()
    return model


def train(env, p):
    set_seed(env, p["seed"])
    model = create_model(env, p["layers"], p["learning_rate"])

    # Setup filewriter
    timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    fn = timestr + "_" + p["name"]
    logdir = "logs/" + fn
    checkpointdir = "checkpoints/" + fn
    file_writer = tf.summary.create_file_writer(logdir)
    file_writer.set_as_default()

    # Log hyperparameters
    hyperparameters = [tf.convert_to_tensor([k, str(v)]) for k, v in p.items()]
    tf.summary.text("hyperparameters", tf.stack(hyperparameters), 0)

    # Train
    dqn(env, model, p["gamma"], p["epsilon"], p["epsilon_decay"],
        p["epsilon_min"], p["episodes"], p["buffer_size"], p["batch_size"],
        p["target_update_freq"], p["checkpoint_freq"],
        checkpoint_path=checkpointdir)

    return model


def play(env, model, episodes=10):
    for i in range(episodes):
        done = False
        state = env.reset()
        steps = 1
        ret = 0
        while not done:
            Qs = model.predict(np.array([state]))[0]
            action = np.argmax(Qs)
            state, reward, done, info = env.step(action)
            env.render()
            steps += 1
            ret += reward
            print("\rEpisode {}: Step {}, Return {}".format(i, steps, ret),
                end="")
        print("")


def render(env, model, fn, episodes=10):
    # The resulting gif is quite large
    # The size can be significantly reduced with gifsicle
    # $ gifsicle -O3 --lossy=30 -o output.gif input.gif
    from PIL import Image
    import PIL.ImageDraw as ImageDraw
    import imageio
    imgs = []
    for i in range(episodes):
        done = False
        state = env.reset()
        steps = 1
        ret = 0
        while not done:
            Qs = model.predict(np.array([state]))[0]
            action = np.argmax(Qs)
            state, reward, done, info = env.step(action)

            img = env.render(mode="rgb_array")
            im = Image.fromarray(img)
            drawer = ImageDraw.Draw(im)
            drawer.text((im.size[0]/20,im.size[1]/18),
                "Episode {}: Step {}, Return {}".format(i, steps, ret),
                fill=(0,0,0))
            imgs.append(im)

            steps += 1
            ret += reward
            print("\rEpisode {}: Step {}, Return {}".format(i, steps, ret),
                end="")
        print("")

    imageio.mimwrite(fn, imgs, fps=30)


def hyperparameter_search(env):
    list_gamma = [0.9,0.95,0.99]
    list_epsilon_decay = [0.8,0.9,0.95]
    list_target_update_freq = [1,10,20,50]

    for gamma in list_gamma:
        for epsilon_decay in list_epsilon_decay:
            for target_update_freq in list_target_update_freq:
                for seed in range(5):
                    p = {
                        "seed": seed,
                        "gamma": gamma,
                        "epsilon": 1,
                        "epsilon_decay": epsilon_decay,
                        "epsilon_min": 0.01,
                        "episodes": 100,
                        "buffer_size": 2000,
                        "batch_size": 32,
                        "target_update_freq": target_update_freq,
                        "checkpoint_freq": 100,
                        "layers": [24,24],
                        "learning_rate": 1e-3,
                        "name": (str(gamma) + "-" + str(epsilon_decay) + "-"
                            + str(target_update_freq)) + "_" + str(seed)
                    }
                    train(env, p)


env = gym.make("CartPole-v0")
p = {
    "seed": 1,
    "gamma": 0.9,
    "epsilon": 1,
    "epsilon_decay": 0.95,
    "epsilon_min": 0.01,
    "episodes": 100,
    "buffer_size": 2000,
    "batch_size": 32,
    "target_update_freq": 50,
    "checkpoint_freq": 100,
    "layers": [24,24],
    "learning_rate": 1e-3,
    "name": "test"
}


#### Hyperparameter search ####
#hyperparameter_search(env)


#### Train ####
model = train(env, p)

#### Play ####
#model = create_model(env, p["layers"], p["learning_rate"])
#model.load_weights("checkpoints/20210424-165027_test/weights-00000000-00000000")
#play(env, model)

### Render ####
#model = create_model(env, p["layers"], p["learning_rate"])
#model.load_weights("checkpoints/20210424-165027_test/weights-00000000-00000000")
#render(env, model, "cartpole_dqn.gif", episodes=10)
