############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
from collections import deque
from torch.autograd import Variable

class ReplayBuffer:

    def __init__(self,capacity):
        self.capacity = capacity
        self.position = 0
        self.collections_deque = deque(maxlen = capacity)

    def add_memory(self, transition):
        if len(self.collections_deque) < self.capacity:
            self.collections_deque.append(None)
        self.collections_deque[self.position] = transition
        self.position = (self.position+1)%self.capacity

    def random_sample(self, batch_size):
        indices = np.arange(len(self.collections_deque))
        rnd_indices = np.random.choice(indices, size = batch_size)
        return np.array(self.collections_deque)[rnd_indices.astype(int)]

    def __len__(self):
        return len(self.collections_deque)

# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.layer_3 = torch.nn.Linear(in_features = 100, out_features = 100)
        self.layer_4 = torch.nn.Linear(in_features = 100, out_features = 100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        layer_4_output = torch.nn.functional.relu(self.layer_4(layer_3_output))
        output = self.output_layer(layer_4_output)
        return output

class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)

        self.target_network = Network(input_dimension = 2, output_dimension = 4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    def weights():
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length (you will need to increase this)
        # Reset the total number of steps which the agent has taken
        self.episode_length = 1000
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        print(self.num_steps_taken)
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None

        self.buffer = ReplayBuffer(1000)

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Here, the action is random, but you can change this
        # Action I have chosen is epsilon-greedy implementation
        dqn = DQN()
        delta = 0.0001
        start_time = time.time()
        end_time = start_time + 600

        eps_decay = 50
        if self.num_steps_taken % self.episode_length == 0: #self.num_steps_taken == 0:
            self.episode_length = max(100,self.episode_length - eps_decay)
        # else:
        #     self.episode_length = max(100,self.episode_length - eps_decay)

        print(self.num_steps_taken)
        print(self.episode_length)

        if self.num_steps_taken == 0:
            self.epsilon = 1
        else:
            self.epsilon = max(self.epsilon - delta, 0)

        print(self.epsilon)

        #action = np.random.uniform([0,1,2,3])

        # put a delta in to make epsilon decay so eventually it only takes maximal thing - so no randomness

        if state is None:
            action = np.random.choice([0,1,2,3])
        else:
            state_tensor = torch.from_numpy(state)
            sample = np.random.choice(range(1,100))/100
            if sample >= self.epsilon:
                with torch.no_grad():
                    action = np.argmax(dqn.q_network(state_tensor))
                    print(action)
            else:
                if 0.05 < sample < self.epsilon:
                    action = np.random.choice([0,2,3])
                else:
                    action = np.random.choice([0,1,2,3])

        # Original action given is below and it is continous !! :
        #action = np.random.uniform(low=-0.01, high=0.01, size=2).astype(np.float32)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action

        continuous_action = self._discrete_action_to_continuous(action)

        return continuous_action

    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:  # Move right
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        elif discrete_action == 1:
            continuous_action = np.array([-0.01,0], dtype = np.float32)
        elif discrete_action == 2:
            continuous_action = np.array([0,-0.02], dtype = np.float32)
        else:
            continuous_action = np.array([0,0.02], dtype = np.float32)
        return continuous_action
    # Function to set the next state and distance, which resulted from applying action self.action at state self.state

    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward = (1 - distance_to_goal)**2
        print(reward)
        # Create a transition
        transition = (self.state, self.action, reward, next_state)

        #print(transition)
        # Now you can do something with this transition ...
        self.buffer.add_memory(transition)
        print(len(self.buffer))

        #self.optimise_model()
        BATCH_SIZE = 500
        if len(self.buffer) < BATCH_SIZE:
            return

        dqn = DQN()
        network = Network(input_dimension=2, output_dimension=4)

        transitions = self.buffer.random_sample(BATCH_SIZE)

        state_batch = np.zeros((BATCH_SIZE,2))
        next_state_batch = np.zeros((BATCH_SIZE,2))
        action_batch = np.zeros((BATCH_SIZE,1))
        reward_batch = np.zeros((BATCH_SIZE,1))

        #print(transitions)
        #print(transitions[1][0],transitions[2][0])
        for i in range(BATCH_SIZE):
            state_batch[i][0] = transitions[i][0][0]
            state_batch[i][1] = transitions[i][0][1]
            action_batch[i][0] = transitions[i][1]
            reward_batch[i][0] = transitions[i][2]
            next_state_batch[i][0] = transitions[i][3][0]
            next_state_batch[i][1] = transitions[i][3][1]

        state_batch_tensor = torch.tensor(state_batch).float()
        #print(state_batch_tensor)
        action_batch_tensor = torch.tensor(action_batch).long()
        reward_batch_tensor = torch.tensor(reward_batch).float()
        next_state_batch_tensor = torch.tensor(next_state_batch).float()
        #print(next_state_batch_tensor)

        network_prediction = dqn.q_network.forward(state_batch_tensor).gather(1, action_batch_tensor).unsqueeze(1)
        #print(network_prediction)

        next_state_values = dqn.target_network.forward(next_state_batch_tensor).max(1)[0].detach()

        gamma = 0.7
        expected_state_values = (next_state_values)*gamma + reward_batch_tensor
        expected_state_values = expected_state_values.unsqueeze(1)
        #print(expected_state_values)

        # Update Loss

        loss = torch.nn.MSELoss()(network_prediction, expected_state_values)

        dqn.optimiser.zero_grad()
        loss.backward()

        dqn.optimiser.step()
        loss_value = loss.item()
        print(f'loss value = {loss_value}')
        if self.num_steps_taken % self.episode_length == 0:
            dqn.target_network.load_state_dict(dqn.q_network.state_dict())

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        dqn = DQN()
        state_tensor = torch.tensor(state).unsqueeze(0) #np.argmax(dqn.q_network(state_tensor))
        with torch.no_grad():
            action = np.argmax(dqn.q_network(state_tensor))
        continuous_action = self._discrete_action_to_continuous(action)

        return continuous_action

    #
    # def optimise_model(self):
    #     BATCH_SIZE = 1000
    #     buffer = ReplayBuffer(1000000)
    #     if len(buffer) < BATCH_SIZE:
    #         return
    #
    #     dqn = DQN()
    #     network = Network(input_dimension=2, output_dimension=4)
    #
    #     transitions = buffer.random_sample(BATCH_SIZE)
    #
    #     state_batch = np.zeros((BATCH_SIZE,2))
    #     next_state_batch = np.zeros((BATCH_SIZE,2))
    #     action_batch = np.zeros((BATCH_SIZE,1))
    #     reward_batch = np.zeros((BATCH_SIZE,1))
    #
    #     #print(transitions)
    #     #print(transitions[1][0],transitions[2][0])
    #     for i in range(BATCH_SIZE):
    #         state_batch[i][0] = transitions[i][0][0]
    #         state_batch[i][1] = transitions[i][0][1]
    #         action_batch[i][0] = transitions[i][1]
    #         reward_batch[i][0] = transitions[i][2]
    #         next_state_batch[i][0] = transitions[i][3][0]
    #         next_state_batch[i][1] = transitions[i][3][1]
    #
    #     state_batch_tensor = torch.tensor(state_batch).float()
    #     action_batch_tensor = torch.tensor(action_batch).long()
    #     reward_batch_tensor = torch.tensor(reward_batch).float()
    #     next_state_batch_tensor = torch.tensor(next_state_batch).float()
    #
    #     network_prediction = dqn.q_network.forward(state_batch_tensor).gather(1,action_batch_tensor).unsqueeze(1)
    #     #print(network_prediction)
    #
    #     next_state_values = dqn.target_network.forward(next_state_batch_tensor).max(1)[0].detach()
    #     #print(next_state_values)
    #
    #     gamma = 0.9
    #     expected_state_values = (next_state_values)*gamma + reward_batch_tensor
    #     expected_state_values = expected_state_values.unsqueeze(1)
    #
    #     # Update Loss
    #
    #     loss = torch.nn.MSELoss()(network_prediction, expected_state_values)
    #
    #     dqn.optimiser.zero_grad()
    #     loss.backward()
    #
    #     dqn.optimiser.step()
    #     loss_value = loss.item()
    #     if self.num_steps_taken % self.episode_length == 0:
    #         dqn.target_network.load_state_dict(dqn.q_network.state_dict())

        #return loss_value

# from collections import deque
#
# class ReplayBuffer:
#
#     def __init__(self,capacity):
#         self.capacity = capacity
#         self.position = 0
#         self.collections_deque = deque(maxlen = capacity)
#
#     def add_memory(self, transition):
#         if len(self.collections_deque) < self.capacity:
#             self.collections_deque.append(None)
#         self.collections_deque[self.position] = transition
#         self.position = (self.position+1)%self.capacity
#
#     def random_sample(self, batch_size):
#         indices = np.arange(len(self.collections_deque))
#         rnd_indices = np.random.choice(indices, size = batch_size)
#         return np.array(self.collections_deque)[rnd_indices.astype(int)]
#
#     def __len__(self):
#         return len(self.collections_deque)
#
# # The Network class inherits the torch.nn.Module class, which represents a neural network.
# class Network(torch.nn.Module):
#
#     # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
#     def __init__(self, input_dimension, output_dimension):
#         # Call the initialisation function of the parent class.
#         super(Network, self).__init__()
#         # Define the network layers. This example network has two hidden layers, each with 100 units.
#         self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
#         self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
#         self.layer_3 = torch.nn.Linear(in_features = 100, out_features = 100)
#         self.layer_4 = torch.nn.Linear(in_features = 100, out_features = 100)
#         self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)
#
#     # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
#     def forward(self, input):
#         layer_1_output = torch.nn.functional.relu(self.layer_1(input))
#         layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
#         layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
#         layer_4_output = torch.nn.functional.relu(self.layer_4(layer_3_output))
#         output = self.output_layer(layer_4_output)
#         return output
#
# class DQN:
#
#     # The class initialisation function.
#     def __init__(self):
#         # Create a Q-network, which predicts the q-value for a particular state.
#         self.q_network = Network(input_dimension=2, output_dimension=4)
#
#         self.target_network = Network(input_dimension = 2, output_dimension = 4)
#         # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
#         self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
#
#     def weights():
#         self.target_network.load_state_dict(self.q_network.state_dict())
#         self.target_network.eval()
