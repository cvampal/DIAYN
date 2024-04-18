import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy, Discriminator
import numpy as np

class DIAYN(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.num_skill = args.num_skills
        
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.num_inputs = num_inputs
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.p_z = torch.full((self.num_skill,), 1/self.num_skill).to(self.device)
        self.p_zs = torch.tile(self.p_z, (args.batch_size,1)).to(self.device)

        self.critic = QNetwork(num_inputs+self.num_skill, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs+self.num_skill, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.discriminator = Discriminator(num_inputs, self.num_skill, args.hidden_size)
        self.dis_optim = Adam(self.discriminator.parameters(), lr=args.lr)
        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs+self.num_skill, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs+self.num_skill, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def sample_skill(self):
        probs = self.p_z.numpy()
        skill = np.random.choice(self.num_skill, p=probs)
        return skill

    def concat_ss(self, state, skill):
        skill_one_hot = np.zeros((self.num_skill,))
        skill_one_hot[skill] = 1.0
        return np.concatenate([state, skill_one_hot])

    def update_critic(self, batch):
        state_batch, action_batch, next_state_batch, mask_batch = batch
        states, zs_one_hot = torch.split(next_state_batch, [self.num_inputs, self.num_skill], dim=-1)
        zs = zs_one_hot.argmax(-1)
        with torch.no_grad():
            logits = self.discriminator(states)
            rewards = -F.cross_entropy(logits, zs, reduction='none')
            log_p_z = torch.log((self.p_zs * zs_one_hot).sum(-1) + 1e-6)
            rewards -= log_p_z
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            
        next_q_value = rewards.view(-1,1) - (1 - mask_batch) * self.gamma * (min_qf_next_target)
        
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        #print(rewards.shape,min_qf_next_target.shape,next_q_value.shape)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        return qf1_loss.item(), qf2_loss.item()

    def update_actor(self, state_batch):
        action, log_action, _ = self.policy.sample(state_batch)
        with torch.no_grad():
            qf1_pi, qf2_pi = self.critic(state_batch, action)
            min_qf = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_action) - min_qf).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        return policy_loss.item(), log_action.detach()


    def update_discriminator(self, next_state_batch, skill_batch):
        states, _ = torch.split(next_state_batch, [self.num_inputs, self.num_skill], dim=-1)
        dis_logits = self.discriminator(states)
        dis_loss = F.cross_entropy(dis_logits, skill_batch)

        self.dis_optim.zero_grad()
        dis_loss.backward()
        self.dis_optim.step()
        return dis_loss.item()


    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, skill_batch = memory.sample(batch_size=batch_size)

        #reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        skill_batch = torch.LongTensor(skill_batch).to(self.device)

        
        #update critic network
        qf1_loss, qf2_loss = self.update_critic((state_batch, action_batch, next_state_batch, mask_batch))

        #update policy network
        policy_loss, log_pi = self.update_actor(state_batch)

        #update discriminator
        dis_loss = self.update_discriminator(next_state_batch, skill_batch)

        #print(dis_loss)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy)).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha_tlogs, dis_loss

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'discriminator_optim_state_dict': self.dis_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.dis_optim.load_state_dict([checkpoint['discriminator_optim_state_dict']])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
                self.discriminator.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
                self.discriminator.eval()

