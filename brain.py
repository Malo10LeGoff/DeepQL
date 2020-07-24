# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 21:59:45 2020

@author: LENOVO
"""

### Creation of the brain which takes the states as inputs and returns the Q-values of the actions as output

### Creation of the brain
import tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class NN(object):
    def __init__(self, nb_actions = 5, lr = 0.001):
        self.learning_rate = lr
        
        inp = Input(shape =(3,))
        layer1 = Dense(units = 64, activation = 'sigmoid')(inp)
        dropout1 = Dropout(rate = 0.1)(layer1)
        layer2 = Dense(units = 32, activation = 'sigmoid')(dropout1)
        dropout2 = Dropout(rate = 0.1)(layer2)
        output_layer = Dense(units = nb_actions, activation = 'softmax')(dropout2)  ### No activation at all because it would maybe change the value of Q for instance relu doesn't allow negative numbers
        
        ### Assembling the architecture of the nn under an object model
        self.model = Model(inp, output_layer)

        
        self.model.compile(optimizer = Adam(lr = self.learning_rate), loss= 'mse')

        
        