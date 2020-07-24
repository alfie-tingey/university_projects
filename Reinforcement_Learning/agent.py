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
import time
#CID = 156819
torch.manual_seed(int(time.time()))

class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length (you will need to increase this)
        # Reset the total number of steps which the agent has taken
        self.episode_length = 2000
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        #print(self.num_steps_taken)
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None

        self.distance_to_goal_list = []

        # Initialise Experience replay buffer

        self.buffer = ReplayBuffer(15000)

        self.epsilon = 1

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

        # Choose a delta so that epsilon will decay to zero in time
        delta = 0.00005

        # if self.num_steps_taken == 0:
        #     self.epsilon = 1
        self.epsilon = self.epsilon - delta
        self.epsilon = max(self.epsilon, 0)

        #delta = 0.09
        eps_decay = 180
        if (self.num_steps_taken) % self.episode_length == 0: #self.num_steps_taken == 0:
            self.episode_length = max(500,self.episode_length - eps_decay)
            #self.epsilon = max(self.epsilon - delta, 0)
            # Update weights of Q network
            dqn.target_network.load_state_dict(dqn.q_network.state_dict())

        #self.epsilon = max(self.epsilon - delta, 0)

        # print(self.num_steps_taken)
        # print(self.episode_length)
        # print(self.epsilon)

        #action = np.random.uniform([0,1,2,3])

        # put a delta in to make epsilon decay so eventually it only takes maximal thing - so no randomness

        if state is None:
            action = np.random.choice([0,1,2,3])
        else:
            state_tensor = torch.from_numpy(state).float()
            #print(state_tensor)
            #state_tensor = Variable(state).float()

            # Epsilon Greedy like exploration method.
            sample = np.random.choice(range(1,100))/100
            if sample >= self.epsilon:
                with torch.no_grad():
                    _, action = torch.max(dqn.q_network(state_tensor),0)
            else:
                # Make the movement left pess likely
                if 0.1 < sample < self.epsilon:
                    action = np.random.choice([0,2,3])
                else:
                    action = np.random.choice([0,1,2,3])
        #print(action)

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
            continuous_action = np.array([-0.02,0], dtype = np.float32)
        elif discrete_action == 2:
            continuous_action = np.array([0,-0.02], dtype = np.float32)
        else:
            continuous_action = np.array([0,0.02], dtype = np.float32)
        return continuous_action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state

    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward

        distance_to_goal_list = []
        distance_to_goal_list.append(distance_to_goal)

        if distance_to_goal < 0.1:
            reward = (1-distance_to_goal)*1
        elif 0.1 <= distance_to_goal < 0.3:
            reward = (1 - distance_to_goal)*0.2
        elif 0.3 <= distance_to_goal < 0.5:
            reward = (1 - distance_to_goal)*0.1
        elif 0.5 <= distance_to_goal <= 0.7 and distance_to_goal_list[-1] < distance_to_goal:
            reward = (1-distance_to_goal)*0.07
        elif 0.5 <= distance_to_goal <= 0.7 and not distance_to_goal_list[-1] < distance_to_goal:
            reward = (1-distance_to_goal)*0.05
        elif 0.7 < distance_to_goal <= 0.8:
            reward = (1 - distance_to_goal)*0.04
        else:
            if self.action == 0:
                reward = (1 - distance_to_goal)*0.04
            elif self.action == 1:
                reward = -1
            else:
                reward = (1 - distance_to_goal)*0.01

        distance_to_goal_list.append(distance_to_goal)

        # Create a transition
        transition = (self.state, self.action, reward, next_state)

        #print(transition)
        # Now you can do something with this transition ...

        # Add transition to replay buffer.
        self.buffer.add_memory(transition)

        # Train model using optimisation function. Batch size of 500 used.
        self.optimise_model(500)


    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        dqn = DQN()
        state_tensor = torch.from_numpy(state).float()
        with torch.no_grad():
            _, action = torch.max(dqn.q_network(state_tensor),0)
            #print(action)
        # Make action continuous so agent can move
        continuous_action = self._discrete_action_to_continuous(action)

        return continuous_action

    def optimise_model(self, batch_size):

        BATCH_SIZE = batch_size
        # Don't train when amount of transitions is less than batch_size
        if len(self.buffer) < BATCH_SIZE:
            return

        dqn = DQN()
        network = Network(input_dimension=2, output_dimension=4)

        transitions = self.buffer.random_sample(BATCH_SIZE)

        # Set tensors and take transitions from the replay buffer
        transitions = np.array(transitions).transpose()
        state_batch = np.vstack(transitions[0])
        #print(state_batch)
        action_batch = np.vstack(transitions[1])
        reward_batch = np.vstack(transitions[2])
        next_state_batch = np.vstack(transitions[3])

        state_batch_tensor = torch.tensor(state_batch).float()

        action_batch_tensor = torch.tensor(action_batch).long()
        reward_batch_tensor = torch.tensor(reward_batch).float()
        next_state_batch_tensor = torch.tensor(next_state_batch).float()


        network_prediction = dqn.q_network.forward(state_batch_tensor).gather(1, action_batch_tensor)
        next_state_values = dqn.target_network.forward(next_state_batch_tensor).max(1)[0].detach()

        # initialise gamma
        gamma = 0.9
        # Create input into loss function
        expected_state_values = (next_state_values)*gamma + reward_batch_tensor
        expected_state_values = expected_state_values.unsqueeze(1)

        # This is so the input and output in the loss function have the same shape
        expected_state_values = expected_state_values[:,:,-1]


        loss = torch.nn.SmoothL1Loss()(network_prediction, expected_state_values)
        #print((network_prediction.shape),(expected_state_values.shape))

        # Train the agent
        dqn.optimiser.zero_grad()
        loss.backward()
        if self.num_steps_taken < 2000:
            pass
        else:
            dqn.optimiser.step()
            loss_value = loss.item()
            #print(f'loss value = {loss_value}')

class ReplayBuffer:

    def __init__(self,capacity):
        # Set parameters
        self.capacity = capacity
        self.position = 0
        self.collections_deque = deque(maxlen = capacity)

    def add_memory(self, transition):
        # Define function to add a transition into the buffer
        if len(self.collections_deque) < self.capacity:
            self.collections_deque.append(None)
        self.collections_deque[self.position] = transition
        self.position = (self.position+1)%self.capacity

    def random_sample(self, batch_size):
        # generate a random mini batch
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
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.0001)
