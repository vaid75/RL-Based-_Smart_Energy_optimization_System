import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# SETTINGS
BATCH_SIZE       = 64
LR               = 0.005
GAMMA            = 0.99
TARGET_UPDATE    = 50
MEMORY_CAPACITY  = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# network
class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


# DQN
class DQN:
    def __init__(self, n_states, n_actions):

        self.n_states = n_states
        self.n_actions = n_actions

        self.eval_net = Net(n_states, n_actions).to(device)
        self.target_net = Net(n_states, n_actions).to(device)

        self.memory = np.zeros((MEMORY_CAPACITY, n_states * 2 + 2))
        self.memory_counter = 0
        self.learn_step = 0

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_fn = nn.SmoothL1Loss()   # Huber loss — more stable than MSE


    #  ACTION 
    def choose_action(self, state, epsilon):

        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        if np.random.rand() > epsilon:
            actions = self.eval_net(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.randint(0, self.n_actions)

        return action


    #  STORE 
    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition

        self.memory_counter += 1


    #  LEARN 
    def learn(self):

        if self.memory_counter < BATCH_SIZE:
            return 0.0

        if self.learn_step % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step += 1

        sample_index = np.random.choice(min(self.memory_counter, MEMORY_CAPACITY), BATCH_SIZE)
        batch = self.memory[sample_index, :]

        b_s = torch.FloatTensor(batch[:, :self.n_states]).to(device)
        b_a = torch.LongTensor(batch[:, self.n_states:self.n_states+1].astype(int)).to(device)
        b_r = torch.FloatTensor(batch[:, self.n_states+1:self.n_states+2]).to(device)
        b_s_ = torch.FloatTensor(batch[:, -self.n_states:]).to(device)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()

        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    # SAVE & LOAD
    def save_model(self, path="dqn_model.pth"):
        torch.save(self.eval_net.state_dict(), path)

    def load_model(self, path="dqn_model.pth"):
        self.eval_net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.eval_net.eval()
        self.target_net.eval()