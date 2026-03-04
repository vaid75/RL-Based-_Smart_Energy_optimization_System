import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Basic Hyperparameters
# -----------------------------
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.9              # reward discount factor
MEMORY_SIZE = 2000      # replay buffer size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Neural Network Model
# This network predicts Q-values
# for each possible action
# -----------------------------
class Net(nn.Module):

    def __init__(self, state_size, action_size):

        super(Net, self).__init__()

        # simple feedforward network
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, action_size)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        q_values = self.output(x)

        return q_values


# -----------------------------
# Deep Q Learning Agent
# -----------------------------
class DQN:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        # main network and target network
        self.eval_net = Net(state_size, action_size).to(device)
        self.target_net = Net(state_size, action_size).to(device)

        # replay memory
        self.memory = np.zeros((MEMORY_SIZE, state_size * 2 + 2))
        self.memory_counter = 0

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LEARNING_RATE)
        self.loss_function = nn.MSELoss()


    # -----------------------------
    # Action selection (epsilon-greedy)
    # -----------------------------
    def choose_action(self, state, epsilon):

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # exploration vs exploitation
        if np.random.rand() > epsilon:

            q_values = self.eval_net(state_tensor)

            # choose action with highest Q value
            action_index = torch.argmax(q_values)

            action = action_index.item()

        else:
            # random action for exploration
            action = np.random.randint(0, self.action_size)

        return action


    # -----------------------------
    # Store experience in memory
    # (state, action, reward, next_state)
    # -----------------------------
    def store_transition(self, state, action, reward, next_state):

        transition = np.hstack((state, [action, reward], next_state))

        index = self.memory_counter % MEMORY_SIZE

        self.memory[index] = transition

        self.memory_counter += 1


    # -----------------------------
    # Train the neural network
    # -----------------------------
    def learn(self):

        # sample random batch from memory
        sample_index = np.random.choice(MEMORY_SIZE, BATCH_SIZE)

        batch = self.memory[sample_index]

        states = torch.FloatTensor(batch[:, :self.state_size]).to(device)
        actions = torch.LongTensor(batch[:, self.state_size:self.state_size+1].astype(int)).to(device)
        rewards = torch.FloatTensor(batch[:, self.state_size+1:self.state_size+2]).to(device)
        next_states = torch.FloatTensor(batch[:, -self.state_size:]).to(device)

        # predicted Q values
        q_eval = self.eval_net(states).gather(1, actions)

        # target Q values
        q_next = self.target_net(next_states).detach()

        q_target = rewards + GAMMA * q_next.max(1)[0].view(BATCH_SIZE,1)

        # compute loss
        loss = self.loss_function(q_eval, q_target)

        # update neural network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
