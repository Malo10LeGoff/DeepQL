# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 13:02:00 2020

@author: LENOVO
"""


import numpy as np
import os
import random as rn
from environment import env
from keras.models import load_model

### Setting the hyperparameters

batch_size = 512
max_memory = 3000
epochs = 1000
eps = 0.3
numb_actions = 5
direction_boundary = (numb_actions -1)/2 ### it will be the action corresponding to "do nothing"
temp_incr = 1.5  ### the difference of temperature between each action
### Creation of the environment
env = env(nb_user_ini = 20, data_transfer_ini = 30, starting_month = 0)

### Setting the seed for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

### Loading the pre-trained model

model = load_model('model.h5')

training = False
env.train = training

time_step = 0
state, _, _ = env.observation()
while time_step < 5 * 30 * 24 * 60 :
    
    action = np.argmax(model.predict(state))
    if (action - direction_boundary) < 0:
        direction = -1
    if (action - direction_boundary) > 0:
        direction = 1
    energy_ai = abs(action - direction_boundary) * temp_incr
    month = int(time_step / (30 * 24 * 60))
    next_state, reward, game_over = env.update_env(energy_ai, direction, month)
    time_step += 1
    state = next_state
            
    print("\n")
    print("energy_spent_without_ai : " + str(env.total_energy_noai))
    print("energy_spent_with_ai : " + str(env.total_energy_ai))
