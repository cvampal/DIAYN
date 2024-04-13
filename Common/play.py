import numpy as np
import os
import gymnasium as gym

class Play:
    def __init__(self, env, agent, n_skills):
        if not os.path.exists("Vid/"):
            os.mkdir("Vid/")
        self.env = gym.wrappers.RecordVideo(env, video_folder="Vid/")
        self.agent = agent
        self.n_skills = n_skills
        self.agent.set_policy_net_to_cpu_mode()
        self.agent.set_policy_net_to_eval_mode()


    @staticmethod
    def concat_state_latent(s, z_, n):
        z_one_hot = np.zeros(n)
        z_one_hot[z_] = 1
        if(type(s) == tuple):
            return np.concatenate([s[0], z_one_hot])
        return np.concatenate([s, z_one_hot])

    def evaluate(self):
        for z in range(self.n_skills):
            s = self.env.reset()
            s = self.concat_state_latent(s, z, self.n_skills)
            episode_reward = 0
            for _ in range(self.env.spec.max_episode_steps):
                action = self.agent.choose_action(s)
                s_, r, done, _, _ = self.env.step(action)
                s_ = self.concat_state_latent(s_, z, self.n_skills)
                episode_reward += r
                if done:
                    break
            print(f"skill: {z}, episode reward:{episode_reward:.1f}")
        self.env.close()
