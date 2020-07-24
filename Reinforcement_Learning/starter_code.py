# Import some modules from other libraries
import numpy as np
import torch
import random
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv2
import sys

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
    environment = Environment(display=False, magnification=700)
    # Create an agent
    agent = Agent(environment)
    # Create a DQN (Deep Q-Network)
    dqn = DQN()
    network = Network(input_dimension=2, output_dimension=4)

    #Draw First Graph for online transition
    # loss_vector = []
    # for step_num in range(500):
    #     # Step the agent once, and get the transition tuple for this step
    #     transition = agent.step()
    #     loss = dqn.train_q_network(transition)
    #     loss_vector.append(loss)
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
            print(action_batch[i][0])
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
        #print(loss_value)

        return loss.item()

#
# img = np.zeros((10,10,3),np.uint8)

# cv.rectangle(img,(0,10),(10,0),(511,0,0),3)
# cv.imshow('image',img)

#### This Is the Visualisation for the Optimal States. fucking huge code.

#Create a black image
#     magnification = 700
#     width = 1.0
#     height = 1.0
# # Create an image which will be used to display the environment
#     image = np.zeros([int(magnification * height), int(magnification * width), 3], dtype=np.uint8)
#
#     image.fill(255)
#
#     states = np.zeros(100,dtype=object).reshape(10,10)
#     q_values = []
#     for i in range(10):
#         for j in range(10):
#             states[i,j] = np.around([float(0.05 + 0.1*i) ,float(0.05 + 0.1*j)],decimals = 2)
#
#             initial_state = states[i,j]
#             action_state = [0,1,2,3]
#             for k in range(4):
#                 action = torch.tensor(action_state[k]).long()
#                 continuous_action = agent._discrete_action_to_continuous(action_state[k])
#                 next_state = initial_state + continuous_action
#                 next_state_tensor = torch.tensor(next_state).float()
#                 goal_state = np.array([0.75, 0.85], dtype=np.float32)
#                 distance_to_goal = np.linalg.norm(next_state - goal_state)
#                 reward = agent._compute_reward(distance_to_goal)
#                 reward_state_tensor = torch.tensor(reward)
#                 initial_state_tensor = torch.tensor(initial_state).float()
#                 network_prediction = reward_state_tensor
#                 q_values.append(network_prediction)
#
#     #print(states)
#     q_values = np.array(q_values).reshape(100,4)
#
#     #print(q_values)
#
#     # (0,1,2,3) = (right,left,down,up)
#     max_ind = []
#     min_ind = []
#     max_val = []
#     min_val = []
#     second_ind = []
#     third_ind = []
#     second_index1 = []
#     third_index1 = []
#     for i in range(100):
#         max_ind.append(np.argmax(q_values[i,:]))
#         min_ind.append(np.argmin(q_values[i,:]))
#         q_values_sorted = np.sort(q_values[i,:])
#         second_ind.append(q_values_sorted[1])
#         third_ind.append(q_values_sorted[2])
#         max_val.append(q_values_sorted[3])
#         min_val.append(q_values_sorted[0])
#         second_index = np.where(q_values[i,:] == q_values_sorted[1])
#         third_index = np.where(q_values[i,:]== q_values_sorted[2])
#         for x in second_index[0]:
#             if x != max_ind[i] and x != min_ind[i] and x != q_values_sorted[2]:
#                 second_index = x
#             else:
#                 for r in range(4):
#                     if r != max_ind[i] and r != min_ind[i] and r != q_values_sorted[2]:
#                         second_index = r
#         for y in third_index[0]:
#             if y != max_ind[i] and y != min_ind[i] and y != second_index:
#                 third_index = y
#             else:
#                 for r in range(4):
#                     if r != max_ind[i] and r != min_ind[i] and r != second_index:
#                         third_index = r
#         second_index1.append(second_index)
#         third_index1.append(third_index)
#
#
#     # print(max_ind,min_ind,second_index1,third_index1)
#
#     # for i in range(len(max_ind)):
#     #     print(second_index1[i] == min_ind[i], second_index1[i] == max_ind[i], second_index1[i] == third_index1[i], third_index1[i] == max_ind[i], third_index1[i] == min_ind[i])
#
#     #pts = np.zeros((1,4),dtype = object)
#     for i in range(0,10):
#         for j in range(0,10):
#             pts = np.zeros(4,dtype = object)
#             pts[1] = np.array([[0+70*i,700 - (70*j)],[35+70*i,700 -(35+70*j)],[0+70*i,700 - (70+70*j)]], np.int32)
#             pts[1] = pts[1].reshape((-1,1,2))
#             pts[2] = np.array([[0+70*i,700 - (0+70*j)],[35+70*i,700 - (35+70*j)],[70+70*i,700 - (0+70*j)]], np.int32)
#             pts[2] = pts[2].reshape((-1,1,2))
#             pts[0] = np.array([[70+70*i,700 - (0+70*j)],[35+70*i,700 - (35+70*j)],[70+70*i,700 - (70+70*j)]], np.int32)
#             pts[0] = pts[0].reshape((-1,1,2))
#             pts[3] = np.array([[0+70*i,700 - (70+70*j)],[35+70*i,700 - (35+70*j)],[70+70*i,700 - (70+70*j)]], np.int32)
#             pts[3] = pts[3].reshape((-1,1,2))
#             cv2.fillConvexPoly(image, pts[max_ind[i*9 + j]], (0,255,255))
#             cv2.fillConvexPoly(image, pts[min_ind[i*9 + j]], (255,0,0))
#
#             t1 = (max_val[i*9+j] - second_ind[i*9+j])/(max_val[i*9+j]-min_val[i*9+j])
#             t2 = (max_val[i*9+j] - third_ind[i*9+j])/(max_val[i*9+j]-min_val[i*9+j])
#             A = [255,0,0]
#             B = [0,255,255]
#             C1 = []
#             C2 = []
#             for k in range(3):
#                 C1.append((B[k] - A[k])*t1 + A[k])
#                 C2.append((B[k] - A[k])*t2 + A[k])
#             # C1 = [int(round(x)) for x in C1]
#             # C2 = [int(round(x)) for x in C2]
#             C1 = tuple(C1)
#             C2 = tuple(C2)
#             #C = (B - A)*t + A, where t=(max(Q) - this_Q)/(max(Q) - min(Q))
#             cv2.fillConvexPoly(image, pts[second_index1[i*9+j]], C1)
#             cv2.fillConvexPoly(image, pts[third_index1[i*9+j]], C2)
#
#             cv2.polylines(image,[pts[0]],True,(0,0,0))
#             cv2.polylines(image,[pts[1]],True,(0,0,0))
#             cv2.polylines(image,[pts[2]],True,(0,0,0))
#             cv2.polylines(image,[pts[3]],True,(0,0,0))
#
#     step = 11
#     x = np.linspace(start=0, stop=magnification*height, num=step)
#     y = np.linspace(start=0, stop=magnification*width, num=step)
#
#
#     v_xy = []
#     h_xy = []
#     for i in range(step):
#         v_xy.append( [int(x[i]), 0, int(x[i]), int(magnification*height-1)] )
#         h_xy.append( [0, int(y[i]), int(magnification*width-1), int(y[i])] )
#
#
#     for i in range(step):
#         [x1, y1, x2, y2] = v_xy[i]
#         [x1_, y1_, x2_, y2_] = h_xy[i]
#         cv2.line(image, (x1,y1), (x2, y2), (255,255,255),1 )
#         cv2.line(image, (x1_,y1_), (x2_, y2_), (255,255,255),1 )
#
#     cv2.imshow("Optimal Action Graph", image)
#     cv2.imwrite('optimal_graph_action.png',image)
# # This line is necessary to give time for the image to be rendered on the screen
#     cv2.waitKey(0)

    def max_Q_for_a():
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
        #print(action_batch_tensor)
        reward_batch_tensor = torch.tensor(reward_batch).float()

        network_prediction = dqn.q_network.forward(state_batch_tensor).gather(1,action_batch_tensor).unsqueeze(1)

        #print(network_prediction.detach())

        expected_state_values = reward_batch_tensor.unsqueeze(1)

        return network_prediction.detach()

    # def step():
    #
    # discrete_action = random.choice([0,1,2,3])
    # # Convert the discrete action into a continuous action.
    # continuous_action = agent._discrete_action_to_continuous(discrete_action)
    # # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
    # next_state, distance_to_goal = agent.environment.step(self.state, continuous_action)
    # # Compute the reward for this paction.
    # reward = agent._compute_reward(distance_to_goal)
    # # Create a transition tuple for this step.
    # transition = (self.state, discrete_action, reward, next_state)
    # # Set the agent's state for the next step, as the next state from this step
    # self.state = next_state
    # # Update the agent's reward for this episode
    # self.total_reward += reward
    # # Return the transition
    # return transition

# Draw second graph for experience replay buffer loss vs step
            #max_q = max_Q_for_a()
            #print(max_q)

    magnification = 700
    width = 1.0
    height = 1.0
# Create an image which will be used to display the environment
    image = np.zeros([int(magnification * height), int(magnification * width), 3], dtype=np.uint8)

    image.fill(0)

    for training_iteration in range(25):
        initial_state = np.array([0.15, 0.15], dtype=np.float32)
        transitions1 = []
        for step_num in range(20):
            # Step the agent once, and get the transition tuple for this step

            action_state = [0,1,2,3]
            q_values = []
            for k in range(4):
                action = torch.tensor(action_state[k]).long()
                continuous_action = agent._discrete_action_to_continuous(action_state[k])
                next_state = initial_state + continuous_action
                next_state_tensor = torch.tensor(next_state).float()
                goal_state = np.array([0.75, 0.85], dtype=np.float32)
                distance_to_goal = np.linalg.norm(next_state - goal_state)
                reward = agent._compute_reward(distance_to_goal)
                reward_state_tensor = torch.tensor(reward)
                initial_state_tensor = torch.tensor(initial_state).float()
                network_prediction = reward_state_tensor
                q_values.append(network_prediction.detach())

            max_action = np.argmax(q_values)
            sorted_q_values = np.sort(q_values)
            print(max_action)

            obstacle_space = np.array([[0.3, 0.5], [0.3, 0.6]], dtype=np.float32)

            continuous_action = agent._discrete_action_to_continuous(max_action)
            next_state = initial_state + continuous_action
            if next_state[0] < 0.0 or next_state[0] > 1.0 or next_state[1] < 0.0 or next_state[1] > 1.0:
                next_state = initial_state
                for i in range(len(q_values)):
                    max_action = random.choice([3,2,max_action])
                    continuous_action = agent._discrete_action_to_continuous(max_action)
                    next_state = initial_state + continuous_action

            if obstacle_space[0, 0] <= next_state[0] < obstacle_space[0, 1] and obstacle_space[1, 0] <= next_state[1] < obstacle_space[1, 1]:
                next_state = initial_state
                for i in range(len(q_values)):
                    max_action = random.choice([3,2,max_action])
                    continuous_action = agent._discrete_action_to_continuous(max_action)
                    next_state = initial_state + continuous_action

            print(max_action)
            distance_to_goal = np.linalg.norm(next_state - goal_state)
            reward = agent._compute_reward(distance_to_goal)

            transition = (initial_state, max_action, reward, next_state)
            transitions1.append(transition)

            initial_state = next_state

            buffer.add_memory(transition)
            optimise_model()

    #print(transitions)
    #print(int(transitions[0][0][0]*magnification))
    transition = (next_state,max_action,reward,next_state)
    transitions1.append(transition)
    print(len(transitions1))
    cv2.circle(image,(int(transitions1[0][0][0]*magnification),int((1 - transitions1[0][0][1])*magnification)), 15, (0,0,255), -1)
    cv2.circle(image,(int(transitions1[20][0][0]*magnification),int((1 - transitions1[20][0][1])*magnification)), 15, (0,255,0), -1)
    R = [0,0,255]
    G = [0,255,0]

    for i in range(len(transitions1)):
        t = (20 - i)/(20)
        C1 = []
        for k in range(3):
            C1.append((R[k] - G[k])*t + G[k])
        C = tuple(C1)
        print(C)
        cv2.line(image,(int(transitions1[i][0][0]*magnification),int((1 - transitions1[i][0][1])*magnification)),(int(transitions1[i][3][0]*magnification),int((1-transitions1[i][3][1])*magnification)),C,5)

    step = 11
    x = np.linspace(start=0, stop=magnification*height, num=step)
    y = np.linspace(start=0, stop=magnification*width, num=step)


    v_xy = []
    h_xy = []
    for i in range(step):
        v_xy.append( [int(x[i]), 0, int(x[i]), int(magnification*height-1)] )
        h_xy.append( [0, int(y[i]), int(magnification*width-1), int(y[i])] )


    for i in range(step):
        [x1, y1, x2, y2] = v_xy[i]
        [x1_, y1_, x2_, y2_] = h_xy[i]
        cv2.line(image, (x1,y1), (x2, y2), (255,255,255),1 )
        cv2.line(image, (x1_,y1_), (x2_, y2_), (255,255,255),1 )

    cv2.imshow("Visualisation 2", image)
    #cv2.imwrite('Optimal_action_graph.png',image)
# This line is necessary to give time for the image to be rendered on the screen
    cv2.waitKey(0)

    #
    # Loop over episodes
    # while True:
    #     # Reset the environment for the start of the episode.
    #     agent.reset()
    #     # Loop over steps within this episode. The episode length here is 20.
    #     for step_num in range(500):
    #         # Step the agent once, and get the transition tuple for this step
    #         transition = agent.step()
    #         # loss = dqn.train_q_network(transition)
    #         # print(loss)
    #
    #         buffer.add_memory(transition)
    #         optimise_model()
