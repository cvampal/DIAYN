import gymnasium as gym
from Brain import SACAgent
from utils import Play, Logger, get_params
import numpy as np
from tqdm import tqdm
# import mujoco_py


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros((n,))
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


if __name__ == "__main__":
    params = get_params()

    test_env = gym.make(params["env_name"])
    n_states = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.shape[0]
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})
    print("params:", params)
    test_env.close()
    del test_env, n_states, n_actions, action_bounds

    env = gym.make(params["env_name"],render_mode="rgb_array")

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgent(p_z=p_z, **params)
    logger = Logger(agent, **params)

    
    logger.load_weights()
    player = Play(env, agent, n_skills=params["n_skills"], name=params['env_name'], seed=params['seed'])
    player.dance()
