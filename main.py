# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 23:08:37 2020

@author: LENOVO
"""

### Imports 

import numpy as np
import os
import random as rn
from environment import env
import brain
from DQN import DQN

### Setting the hyperparameters

max_memory = 3000
epochs = 1000
batch_size = 128
eps = 0.3
numb_actions = 5
direction_boundary = (numb_actions -1)/2 ### it will be the action corresponding to "do nothing"
temp_incr = 1.5  ### the difference of temperature between each action
### Creation of the environment
env = env(nb_users_ini = 20, data_transfer_ini = 30, starting_month = 0)
### Creation of the brain
brain = brain.NN(nb_actions = numb_actions)
model = brain.model
### Creation of the memory of the DQN Agent
DQN = DQN()

if(env.train):
    previous_loss = 0
    patience = 0
    for epoch in range(0,epochs):
        loss = 0
        time_step = 0
        game_over = False
        total_reward = 0
        new_month = np.random.randint(0,12)
        env.reset(new_month)
        game_over = env.game_over
        state, _, _ = env.observation()
        while not game_over and time_step < 5 * 30 * 24 * 60 :
            print(time_step)
            ### Choosing the action
            if np.random.rand() < eps:   ### exploration
                action = np.random.randint(0,numb_actions)
                if (action - direction_boundary) < 0:
                    direction = -1
                if (action - direction_boundary) > 0:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temp_incr
            else:
                action = np.argmax(model.predict(state))
                if (action - direction_boundary) < 0:
                    direction = -1
                if (action - direction_boundary) > 0:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temp_incr
            month = new_month + int(time_step / (30 * 24 * 60))
            next_state, reward, game_over = env.update_env(energy_ai, direction, month)
            
            ### Storing the transition
            DQN.remember([state, action, reward, next_state], game_over)
            inputs, targets = DQN.get_batch(model, batch_size)
            loss += model.train_on_batch(inputs, targets)  ### magic method training do the backpropagation and compute the loss
            #print(loss)
            total_reward += reward
            time_step += 1
            state = next_state
         
        print("\n")
        print("Epoch : " + str(epoch))
        print("energy_spent_without_ai : " + str(env.total_energy_noai))
        print("energy_spent_with_ai : " + str(env.total_energy_ai))
        
        model.save("model.h5")
        
        if abs(previous_loss - loss)/loss < 0.01:
            patience += 1
            if patience > 10:
                print("early_stopping")
                break
        else:
            previous_loss = loss
    
    