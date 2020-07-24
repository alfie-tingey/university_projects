# Import some modules from other libraries
import numpy as np
import torch
import random
import matplotlib
import matplotlib.pyplot as plt

# Import the environment module
from environment import Environment


# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment):
        # Set the agent's environment.
        self.environment = environment
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Reset the agent.
        self.reset()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self):
        # Choose an action.
        discrete_action = random.choice([0,1,2,3])
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = 1 - distance_to_goal
        return reward

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:  # Move right
            continuous_action = np.array([0.1, 0], dtype=np.float32)
        elif discrete_action == 1:
            continuous_action = np.array([-0.1,0], dtype = np.float32)
        elif discrete_action == 2:
            continuous_action = np.array([0,-0.1], dtype = np.float32)
        else:
            continuous_action = np.array([0,0.1], dtype = np.float32)
        return continuous_action

class ReplayBuffer:

    def __init__(self,capacity):
        self.capacity = capacity
        self.position = 0
        self.collections_deque = []


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
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transition):
        pass
        # TODO
        # code below is for the online transitions:
        initial_state = transition[0]
        next_state = transition[3]
        action_state = transition[1]
        reward_state = transition[2]
        reward_state_tensor = torch.tensor(reward_state).float()
        #print(reward_state_tensor)
        action_state_tensor = torch.tensor(action_state).long()
        # print(action_state_tensor)
        initial_state_tensor = torch.tensor(initial_state).float()
        next_state_tensor = torch.tensor(next_state)
        # print(initial_state_tensor)
        # print(next_state_tensor)
        network_prediction = self.q_network.forward(initial_state_tensor).gather(0,action_state_tensor).unsqueeze(0)
        #print(network_prediction)
        #transition = (self.state, discrete_action, reward, next_state)
        #print(network_prediction)
        reward_state_tensor = reward_state_tensor.unsqueeze(0)
        #print(reward_state_tensor)
        loss = torch.nn.MSELoss()(network_prediction, reward_state_tensor)
        return loss

# Main entry point
if __name__ == "__main__":

    # Set the random seed for both NumPy and Torch
    # You should leave this as 0, for consistency across different runs (Deep Reinforcement Learning is highly sensitive to different random seeds, so keeping this the same throughout will help you debug your code).
    CID = 156819
    np.random.seed(CID)
    torch.manual_seed(CID)

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop PCs, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=True, magnification=700)
    # Create an agent
    agent = Agent(environment)
    # Create a DQN (Deep Q-Network)
    dqn = DQN()
    network = Network(input_dimension=2, output_dimension=4)

    #Draw First Graph for online transition
    # loss_vector = []
    # for training_iteration in range(25):
    #     agent.reset()
    #     for step_num in range(20):
    #     # Step the agent once, and get the transition tuple for this step
    #         transition = agent.step()
    #         loss = dqn.train_q_network(transition)
    #         loss_vector.append(loss)
    #
    # number_of_steps = range(500)
    # fig, ax = plt.subplots()
    # ax.plot(number_of_steps, loss_vector)
    # ax.set_yscale('log')
    #
    # ax.set(xlabel='Step number', ylabel='Loss value (logarithmic scale)',
    # title='Loss vs Step online transition')
    # ax.grid()
    # plt.show()

    BATCH_SIZE = 50
    buffer = ReplayBuffer(1000000)

    def optimise_model():
        if len(buffer) < BATCH_SIZE:
            return

        transitions = buffer.random_sample(BATCH_SIZE)

        state_batch = np.zeros((BATCH_SIZE,2))
        action_batch = np.zeros((BATCH_SIZE,1))
        reward_batch = np.zeros((BATCH_SIZE,1))

        #print(transitions)
        #print(transitions[1][0],transitions[2][0])
        for i in range(BATCH_SIZE):
            state_batch[i][0] = transitions[i][0][0]
            state_batch[i][1] = transitions[i][0][1]
            action_batch[i][0] = transitions[i][1]
            reward_batch[i][0] = transitions[i][2]

        # Figure out this batch bullshit
        # print(state_batch)
        # print(action_batch)
        # print(reward_batch)

        state_batch_tensor = torch.tensor(state_batch).float()
        action_batch_tensor = torch.tensor(action_batch).long()
        reward_batch_tensor = torch.tensor(reward_batch).float()

        network_prediction = dqn.q_network.forward(state_batch_tensor).gather(1,action_batch_tensor).unsqueeze(1)

        expected_state_values = reward_batch_tensor.unsqueeze(1)

        # Update Loss

        loss = torch.nn.MSELoss()(network_prediction, expected_state_values)

        dqn.optimiser.zero_grad()
        loss.backward()
        dqn.optimiser.step()
        loss_value = loss.item()
        print(loss_value)

        return loss.item()

# Draw second graph for experience replay buffer loss vs step
    # loss_vector = []
    # for training_iteration in range(25):
    #     agent.reset()
    #     for step_num in range(20):
    #     # Step the agent once, and get the transition tuple for this step
    #         transition = agent.step()
    #         buffer.add_memory(transition)
    #         loss = optimise_model()
    #         loss_vector.append(loss)
    #
    # number_of_steps = range(500)
    # fig, ax = plt.subplots()
    # ax.plot(number_of_steps, loss_vector)
    # ax.set_yscale('log')
    #
    # ax.set(xlabel='Step number', ylabel='Loss value (logarithmic scale)',
    # title='Loss vs Step with ReplayBuffer')
    # ax.grid()
    # plt.show()

    #
    # Loop over episodes
    # while True:
    #     # Reset the environment for the start of the episode.
    #     agent.reset()
    #     # Loop over steps within this episode. The episode length here is 20.
    #     for step_num in range(20):
    #         # Step the agent once, and get the transition tuple for this step
    #         transition = agent.step()
    #         # loss = dqn.train_q_network(transition)
    #         # print(loss)
    #
    #         buffer.add_memory(transition)
    #         optimise_model()
