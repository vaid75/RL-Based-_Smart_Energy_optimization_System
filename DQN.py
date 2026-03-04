import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
BATCH_SIZE = 32
LR = 0.001
GAMMA = 0.9
MEMORY_CAPACITY = 2000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Neural Network for Q-values
# -----------------------------
class Net(nn.Module):

    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_states, 16)
        self.fc2 = nn.Linear(16, 16)
        self.out = nn.Linear(16, n_actions)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


# -----------------------------
# Deep Q Learning Agent
# -----------------------------
class DQN:

    def __init__(self, n_states, n_actions):

        self.n_states = n_states
        self.n_actions = n_actions

        self.eval_net = Net(n_states, n_actions).to(device)
        self.target_net = Net(n_states, n_actions).to(device)

        self.memory = np.zeros((MEMORY_CAPACITY, n_states * 2 + 2))
        self.memory_counter = 0

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()


    # -----------------------------
    # Choose action
    # -----------------------------
    def choose_action(self, state, epsilon):

        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        if np.random.rand() > epsilon:

            actions = self.eval_net(state)
            action = torch.argmax(actions).item()

        else:

            action = np.random.randint(0, self.n_actions)

        return action


    # -----------------------------
    # Store experience
    # -----------------------------
    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index] = transition

        self.memory_counter += 1


    # -----------------------------
    # Train network
    # -----------------------------
    def learn(self):

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)

        batch = self.memory[sample_index]

        b_s = torch.FloatTensor(batch[:, :self.n_states]).to(device)
        b_a = torch.LongTensor(batch[:, self.n_states:self.n_states+1].astype(int)).to(device)
        b_r = torch.FloatTensor(batch[:, self.n_states+1:self.n_states+2]).to(device)
        b_s_ = torch.FloatTensor(batch[:, -self.n_states:]).to(device)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()

        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE,1)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
