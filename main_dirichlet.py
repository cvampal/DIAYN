import gymnasium as gym
from agent_dirichlet import SACAgentD
from utils import Play, Logger, get_params
import numpy as np
from tqdm import tqdm
# import mujoco_py


def concat_state_latent(s, z):
    return np.concatenate([s, z])


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

    cst = np.array([0.2]*params["n_skills"])  #alpha = 0.2 for all D
    agent = SACAgentD(cst=cst, **params)
    logger = Logger(agent, **params)

    if params["do_train"]:

        if not params["train_from_scratch"]:
            episode, last_logq_zs = logger.load_weights()
            agent.hard_update_target_network()
            min_episode = episode
            print("Keep training from previous run.")

        else:
            min_episode = 0
            last_logq_zs = 0
            print("Training from scratch.")

        logger.on()
        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            z = agent.sample_z()
            state, _ = env.reset(seed=params['seed'])
            state = concat_state_latent(state, z)
            episode_reward = 0
            logq_zses = []

            max_n_steps = min(params["max_episode_len"], env.spec.max_episode_steps)
            for step in range(1, 1 + max_n_steps):
                #print(state)
                action = agent.choose_action(state)
                next_state, reward, done, done2, _ = env.step(action)
                done = done or done2
                next_state = concat_state_latent(next_state, z)
                agent.store(state, z, done, action, next_state)
                logq_zs = agent.train()
                if logq_zs is None:
                    logq_zses.append(last_logq_zs)
                else:
                    logq_zses.append(logq_zs)
                episode_reward += reward
                state = next_state
                if done:
                    break

            logger.log(episode,
                       episode_reward,
                       0,
                       sum(logq_zses) / len(logq_zses),
                       step
                       )

    else:
        logger.load_weights()
        player = Play(env, agent, n_skills=params["n_skills"])
        player.evaluate()
