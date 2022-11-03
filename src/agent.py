import gym
import time
import numpy as np
import collections
from tensorboardX import SummaryWriter
from environment import environment

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

N = 100
SUCCESS_FRACTION = 0.8
REWARD_BOUND = 90

HIDDEN_LAYER = 128
GAMMA = 0.95
BATCH_SIZE = 8
BUFFER_SIZE = 50000
# Lowering learning rate by factor 1/10 from frozen lake env
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 100
REPLAY_START_SIZE = 2000

EPSILON_STEPS = 5000
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

##TODO
#- env.action_space
#- env.observation_space
#- states as one hot encoded
#- env.action_space.sample()
#- actions as scalars


actions_scalar = {"stay": 0,
                "right":1,
                "left": 2,
                "up":   3,
                "down": 4}

'''
Defining the Deep NN
'''
class DQN(nn.Module):
    ''' Define Deep Neural Network of form input -> hidden -> output
    '''
    def __init__(self, input_nr, action_nr):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(input_nr, HIDDEN_LAYER),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER, action_nr)
        )

    def forward(self, x):
        return self.layers(x.float())

np.random.seed(42)
env = environment(transM='random', xmax=4, ymax=4)
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

def state2one_hot(state):
    res = np.zeros(env.observation_space_n, dtype=np.float32)
    res[state[0]] = 1.0
    return res

class ExperienceBuffer:
    ''' Experience Buffer class with buffer as collections.deque object with length of given capacity
    methods:
        __init__:   initialize the buffer
        __len__:    return the length
        append:     append one given experience and append it on the right
        sample:     sample mini batch from stored experiences and return them as seperated numpy arrays
                    also: delete sampled items (?)
    '''
    # Buffer to store the experiences to learn from
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        # queue one element into the buffer
        self.buffer.append(experience)

    def sample(self, batch_size):
        # sample batch_size elements from the buffer and return the experience
        # as numpy arrays
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    ''' Agent class with environment and experience buffer
    methods:
        __init__:   initialize of environment & buffer
        _reset:     reset of environment and render state if specified (render==True)
        play_step:  make an epsilon greedy step and store it in the experience buffer as specified in text
                    above the code
                    return reward if episode is done else return None
        close:      close the environment
    '''
    def __init__(self, env, exp_buffer, render=False):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset(render)
        #Todo
        self.action_list = [x for x in env.actions]

    def _reset(self, render=False):
        self.state = self.env.reset()
        if render == True:
            self.env.render()

    def play_step(self, net, epsilon=0.0, device="cpu", render=False, flow=False):
        if flow == True:
            action = self.action_list[0]
        elif np.random.random() < epsilon:
            action = self.action_list[np.random.randint(5)]
            #action = self.env.action_space.sample()
        else:
            state_a = np.array([state2one_hot(self.state)])
            state_v = torch.tensor(state_a)
            q_vals_v = net(state_v)
            act_v = torch.max(q_vals_v, dim=1)[1]
            action = self.action_list[act_v.item()]

        # do step in the environment
        new_state, reward, is_done = self.env.step(action)
        if render == True:
            self.env.render()

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            self._reset()
        return reward

    def close(self):
        self.env.close()

def calc_loss(batch, net, tgt_net, device="cpu"):
    ''' Calculate loss according to description above this code
    '''
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor([state2one_hot(s) for s in states])
    next_states_v = torch.tensor([state2one_hot(s) for s in states])
    actions = [actions_scalar[x] for x in actions]
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    done_mask = torch.ByteTensor(dones)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1).long()).squeeze(-1) #.long() for windows?
    ##### ALTERNATIVE loop
    #mat = net(states_v)
    #state_action_values = torch.zeros(mat.shape[0])
    #for i in range(len(actions_v)):
    #    state_action_values[i] = mat[i, actions_v[i]]

    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = rewards_v + GAMMA * next_state_values
    return nn.MSELoss()(state_action_values, expected_state_action_values)
######################################################################

def flowStatistics(tries, net=None, epsilon=None):
    ''' Do "benchmark" of agent with random steps
    @param tries: number of episodes
    @return: av_reward returns average rewards of tries
    '''
    buffer = ExperienceBuffer(BUFFER_SIZE)
    agent = Agent(env, buffer)
    i = 0
    reward_sum = 0

    while i < tries:
        if net == None:
            reward = agent.play_step(net, epsilon, device="gpu", flow=True)
        else:
            reward = agent.play_step(net, epsilon, device="gpu", flow=False)
        reward_sum += reward
        if reward == 100:
            i += 1
    av_reward = reward_sum/tries

    return av_reward

def learn():
    # Try to use gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = DQN(env.observation_space_n, 5)
    tgt_net = DQN(env.observation_space_n, 5)
    net=net.float()
    tgt_net=tgt_net.float()
    writer = SummaryWriter(comment="-DeepQ_w_CustomEnv")
    print(net)

    buffer = ExperienceBuffer(BUFFER_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    start_time = ts
    best_mean_reward = None
    episode_reward = 0

    while True:
        frame_idx += 1
        # Epsilon decay linear or exponentially
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_STEPS)
        #epsilon = EPSILON_FINAL + (EPSILON_START-EPSILON_FINAL)*np.exp(-frame_idx/EPSILON_STEPS)
        reward = agent.play_step(net, epsilon, device="gpu")
        episode_reward += reward

        if reward == 100:
            total_rewards.append(episode_reward)
            last_N = np.array(total_rewards[-N:])
            mean_reward = np.mean(last_N)
            fraction = (last_N > REWARD_BOUND).sum()/N
            print("%d: done %d games, mean reward %.3f, episode reward %d, fraction %.2f, eps %.2f" % (
                frame_idx, len(total_rewards), mean_reward, episode_reward, fraction, epsilon))
            learn_rate = optimizer.state_dict()['param_groups'][0]['lr']
            # writer.add_scalar("learning_rate", learn_rate, frame_idx)
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", episode_reward, frame_idx)

            episode_reward = 0
            if fraction > SUCCESS_FRACTION:
                print("Solved in %d frames!" % frame_idx)
                print('Solved in {} seconds'.format(time.time()-start_time))
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
            print("Target net update, {}".format(frame_idx))

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device="gpu")
        loss_t.backward()
        optimizer.step()

    writer.close()
    return net

if __name__ == '__main__':
    average_reward_benchmark = flowStatistics(100)
    net = learn()
    average_reward_learned = flowStatistics(100, net=net, epsilon=0)

    print('\nBenchmark:\nAverage reward:', average_reward_benchmark, ', Average number of steps: ', 100-average_reward_benchmark)
    print('After Learning:\nAverage reward:', average_reward_learned, ', Average number of steps: ', 100-average_reward_learned)
