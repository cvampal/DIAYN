# from mujoco_py.generated import const
# from mujoco_py import GlfwContext
import cv2
import numpy as np
import os
import json
# GlfwContext(offscreen=True)


class Play:
    def __init__(self, env, agent, n_skills, name, seed):
        self.env = env
        self.agent = agent
        self.n_skills = n_skills
        self.agent.set_policy_net_to_cpu_mode()
        self.agent.set_policy_net_to_eval_mode()
        self.name = name[:-3]
        self.seed = seed
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists(f"Vid/{self.name}/"):
            os.mkdir(f"Vid/{self.name}/")

    @staticmethod
    def concat_state_latent(s, z_, n):
        z_one_hot = np.zeros(n)
        z_one_hot[z_] = 1
        return np.concatenate([s, z_one_hot])

    def evaluate(self):
        data = {}
        for z in range(self.n_skills):
            video_writer = cv2.VideoWriter(f"Vid/{self.name}/skill{z}" + ".avi", self.fourcc, 50.0, (250, 250))
            s, _ = self.env.reset(seed=self.seed)
            s = self.concat_state_latent(s, z, self.n_skills)
            episode_reward = 0
            for _ in range(self.env.spec.max_episode_steps):
                action = self.agent.choose_action(s)
                s_, r, done, done2, _ = self.env.step(action)
                done = done or done2
                s_ = self.concat_state_latent(s_, z, self.n_skills)
                episode_reward += r
                if done:
                    break
                s = s_
                I = self.env.render()
                I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                I = cv2.resize(I, (250, 250))
                video_writer.write(I)
            print(f"skill: {z}, episode reward:{episode_reward:.1f}")
            video_writer.release()
            data[z] = episode_reward
        self.env.close()
        cv2.destroyAllWindows()
        data = dict(sorted(data.items(), key=lambda x: -x[1]))
        print(data)
        json.dump(data, open(f"Vid/{self.name}/skill_reward.json", "w"))



    def dance(self):
        data = {}
        if not os.path.exists(f"Vid/dance/{self.name}/"):
            os.mkdir(f"Vid/dance/{self.name}/")
        for i in range(self.n_skills):
            video_writer = cv2.VideoWriter(f"Vid/dance/{self.name}/dance{i}" + ".avi", self.fourcc, 50.0, (250, 250))
            s, _ = self.env.reset(seed=self.seed)
            
            episode_reward = 0
            for _ in range(self.env.spec.max_episode_steps):
                z = np.random.choice(self.n_skills)
                s = self.concat_state_latent(s, z, self.n_skills)
                action = self.agent.choose_action(s)
                s_, r, done, done2, _ = self.env.step(action)
                done = done or done2
                #s_ = self.concat_state_latent(s_, z, self.n_skills)
                episode_reward += r
                if done:
                    break
                s = s_
                I = self.env.render()
                I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                I = cv2.resize(I, (250, 250))
                video_writer.write(I)
            print(f"dance: {i}, episode reward:{episode_reward:.1f}")
            video_writer.release()
            data[i] = episode_reward
        self.env.close()
        cv2.destroyAllWindows()
        data = dict(sorted(data.items(), key=lambda x: -x[1]))
        print(data)
        json.dump(data, open(f"Vid/dance/{self.name}/dance_reward.json", "w"))
