import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    def __init__(self, state_size, action_size):
        super(Network, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, 196)
        self.bn1 = nn.BatchNorm1d(196)
        self.fc2 = nn.Linear(196, 196)
        self.bn2 = nn.BatchNorm1d(196)
        self.fc3 = nn.Linear(196, 96)
        self.bn3 = nn.BatchNorm1d(96)
        self.fc4 = nn.Linear(96, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc_pi = nn.Linear(32, action_size)
        self.fc_v = nn.Linear(32, 1)

    def pi(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.fc_pi(x)
        return x

    def v(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        v = self.fc_v(x)
        return v

    def get_action(self, s, possible_actions):
        logit = self.pi(torch.from_numpy(s).float().to(device))
        mask = np.ones(self.action_size)
        mask[possible_actions] = 0.0
        logit = logit - 1e8 * torch.from_numpy(mask).float().to(device)
        prob = torch.softmax(logit, dim=-1)

        m = Categorical(prob)
        a = m.sample().item()

        return a


import torch
import torch.optim as optim
import numpy as np

from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, action_size, learning_rate, gamma, lmbda, eps_clip, K_epoch):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.data = []
        self.network = Network(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=5000, gamma=0.9)
        #self.scheduler = OneCycleLR(optimizer=self.optimizer, max_lr=self.learning_rate, total_steps=30000)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, mask_lst, done_lst = [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, mask, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            mask_lst.append(mask)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, prob_a, mask, done = torch.tensor(s_lst, dtype=torch.float).to(device), torch.tensor(a_lst).to(device), \
                                               torch.tensor(r_lst, dtype=torch.float).to(device), torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
                                               torch.tensor(prob_a_lst).to(device), torch.tensor(mask_lst).to(device), torch.tensor(done_lst, dtype=torch.float).to(device)

        self.data = []
        return s, a, r, s_prime, prob_a, mask, done

    def get_action(self, s, possible_actions):
        self.network.eval()
        logit = self.network.pi(torch.from_numpy(s).float().unsqueeze(0).to(device))
        mask = np.ones(self.action_size)
        mask[possible_actions] = 0.0
        logit = logit.squeeze(0) - 1e8 * torch.from_numpy(mask).float().to(device)
        prob = torch.softmax(logit, dim=-1)

        m = Categorical(prob)
        a = m.sample().item()

        return a, prob, mask

    def train(self):
        self.network.train()
        s, a, r, s_prime, prob_a, mask, done = self.make_batch()
        avg_loss = 0.0

        for i in range(self.K_epoch):
            td_target = r + self.gamma * self.network.v(s_prime) * done
            delta = td_target - self.network.v(s)
            delta = delta.cpu().detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)

            logit = self.network.pi(s)
            logit = logit.float() - 1e8 * mask.float()
            pi = torch.softmax(logit, dim=-1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + 0.2 * F.smooth_l1_loss(self.network.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            avg_loss += loss.mean().item()

        return avg_loss / self.K_epoch

    def save_network(self, e, file_dir):
        torch.save({"episode": e,
                    "model_state_dict": self.network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   file_dir + "episode%d.pt" % e)