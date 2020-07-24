# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 22:35:14 2020

@author: LENOVO
"""

### Algorithm DRL itself

import numpy as np

class DQN(object):
    def __init__(self, max_memory = 150, discount = 0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = 0.9
        
    def remember(self, transition, game_over):
        self.memory.append([transition,game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]
            
          
    def get_batch(self, model, batch_size, nb_actions = 5):
        numb_outputs = nb_actions
        numb_inputs = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len(self.memory),batch_size), numb_inputs))  ### During the first iteration the batch size will be bigger than the memory so we have to make smaller batches
        ### to train at the beginning of the episode
        targets = np.zeros((min(len(self.memory),batch_size), numb_outputs))
        random_index = np.random.randint(0, len(self.memory), size = min(batch_size, len(self.memory)))  ### take random indices in the memory
        for i, idx in enumerate(random_index):
            current_state, action,reward, next_state  = self.memory[idx][0]
            game_over = self.memory[idx][1]
            inputs[i] = current_state
            targets[i] = model.predict(next_state[0])
            Qns = max(model.predict(next_state)[0])
            if game_over:
                targets[i,action] = reward
            else:
                targets[i,action] = reward + self.discount * Qns
        return inputs, targets
        