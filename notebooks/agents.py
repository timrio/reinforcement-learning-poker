import os, sys
import gym
import time
import jdc
import numpy as np
import text_flappy_bird_gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from clubs_gym.agent.base import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, player_id, seed = 42):
        self.rand_generator = np.random.RandomState(seed)
        self.player_id = player_id

    def act(self, obs):

        available_chips = obs['stacks'][self.player_id]
        action = self.rand_generator.randint(available_chips)
        return(action)




class QLearningAgent():
    def agent_init(self, agent_init_info):
        self.num_actions = agent_init_info["num_actions"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])

        self.q = {}

        
    def agent_start(self, state):

        current_q = self.q.setdefault(state,[0,0])
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions) # random action selection
        else:
            action = self.argmax(current_q) # greedy action selection
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_step(self, reward, state):
        # Choose action using epsilon greedy.
        current_q = self.q.setdefault(state,[0,0])
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        previous_values_list = self.q[self.prev_state]
        previous_values_list[self.prev_action] += self.step_size*(reward + self.discount*np.max(self.q[state]) - self.q[self.prev_state][self.prev_action])
        self.q[self.prev_state] = previous_values_list
        
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_end(self, reward):
        previous_values_list = self.q[self.prev_state]
        previous_values_list[self.prev_action] += self.step_size*(reward - self.q[self.prev_state][self.prev_action])
        self.q[self.prev_state] = previous_values_list

        
    def argmax(self, q_values):
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)

    def set_epsilon(self, value):
        self.epsilon = value

    @staticmethod
    def load(path):
        obj = pickle.load(open(path,'wb'))
        return(obj)