import argparse
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
from sac import SAC
from diayn import DIAYN
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory, ReplayMemoryZ

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Without Rewards (DIAYN) Args')

parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--num_skills', type=int, default=8, metavar='N',
                    help='number of skills to learn (default: 8)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
test_env = gym.make(args.env_name, render_mode="rgb_array")
#func = lambda n: n%10 == 0
#test_env = gym.wrappers.RecordVideo(env=test_env, video_folder=f"./demo/{args.env_name}/", name_prefix="test-video", episode_trigger=func)
#env.seed(args.seed)
#env.action_space.seed(args.seed)
_  = test_env.reset()
#test_env.start_video_recorder()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = DIAYN(env.observation_space.shape[0], env.action_space, args)
memory = ReplayMemoryZ(args.replay_size, args.seed)
writer = SummaryWriter('runs/{}_DIAYN_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                            args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Training Loop
total_numsteps = 0
updates = 0
train_rewards = []
eval_rewards = []

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    z = agent.sample_skill()
    state, _ = env.reset(seed=args.seed)
    state = agent.concat_ss(state, z)
    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, dis_loss = agent.update_parameters(memory, args.batch_size, updates)
                writer.add_scalar('loss/discriminator_loss', dis_loss, updates)
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('loss/discriminator_loss', dis_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, terminated, truncated, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        done = terminated or truncated
        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        next_state = agent.concat_ss(next_state, z)
        memory.push(state, action, reward, next_state, mask, z) # Append transition to memory
        state = next_state

    if total_numsteps > args.num_steps:
        break
    train_rewards.append(episode_reward)
    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state, _ = test_env.reset(seed=args.seed)
            episode_reward = 0
            done = False
            state = agent.concat_ss(state, z)
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, done2,  _ = test_env.step(action)
                done = done or done2
                episode_reward += reward
                #test_env.render()


                state = agent.concat_ss(next_state,z)
            avg_reward += episode_reward
        avg_reward /= episodes
        eval_rewards.append(avg_reward)
        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")
#test_env.close_video_recorder()
test_env.close()
env.close()

torch.save(torch.tensor(train_rewards), f"logs/{args.env_name}_train_reward_{args.seed}.pt")
torch.save(torch.tensor(eval_rewards), f"logs/{args.env_name}_eval_rewards_{args.seed}.pt")