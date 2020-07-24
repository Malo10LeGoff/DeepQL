# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 21:20:10 2020

@author: LENOVO
"""
### Imports

import pandas as pd
import numpy as np
import random

### defining the nevironment

class env():
    def __init__(self, nb_users_ini, data_transfer_ini, starting_month, optimal_temperature = (18.0, 24.0)):
        self.tmin = -20
        self.tmax = 80
        self.min_users = 10
        self.max_users = 100
        self.variation_user_per_minutes = 5
        self.min_rate_data_transfer = 20
        self.max_rate_data_transfer = 300
        self.data_transfer_variation = 10
        self.nb_user = nb_users_ini
        self.data_transfer = data_transfer_ini
        self.initial_month = starting_month
        self.nb_users_ini = nb_users_ini
        self.data_transfer_ini = data_transfer_ini
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.monthly_atmospheric_temperature = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 23.0, 24.0, 22.0, 10.0, 5.0, 1.0]
        self.month = starting_month
        self.atmoshperic_temperature = self.monthly_atmospheric_temperature[self.month]
        self.current_T = self.atmoshperic_temperature + 1.25 * (self.nb_user + self.data_transfer)
        self.temperature_noai = (optimal_temperature[0] + optimal_temperature[1])/2.0  ### starting temperature for the system without ai
        self.temperature_ai = self.current_T ### starting temperature for the system with ai
        self.optimal_temperature = optimal_temperature
        self.train = 1
        
        
    ### Creating the method that, after an action has been taken by the agent (so a variation of temperature), update the environment
    def update_env(self, energy_ai, direction, month):  ### combining these 2 parameters (energy_ai and direction), we have our action. Month is for updating the monthly temperature
        
    
        ### Getting the reward, so computing the action taken by the system with no ai and comparing it with the action of the agent
        energy_noai = 0
        if self.temperature_noai < self.optimal_temperature[0]:
            energy_noai = abs(self.temperature_noai - self.optimal_temperature[0])
            self.total_energy_noai += energy_noai
            
        if self.temperature_noai > self.optimal_temperature[1]:
            energy_noai = abs(self.temperature_noai - self.optimal_temperature[1])
            self.total_energy_noai+= energy_noai
        self.reward = energy_noai - energy_ai
        self.reward = 0.001 * self.reward
        
        ### Going to the next state
        self.atmoshperic_temperature = self.monthly_atmospheric_temperature[month]
        
        self.data_transfer += random.randint(-10,10)
        if self.data_transfer < 20:
            self.data_transfer = 20            
        if self.data_transfer > 300:
            self.data_transfer = 300
            
        self.nb_user += random.randint(-5,5)
        if self.nb_user < 10:
            self.nb_user = 10            
        if self.nb_user > 100:
            self.nb_user = 100
            
        past_temperature = self.current_T
        self.current_T = self.atmoshperic_temperature + 1.25 * (self.nb_user + self.data_transfer)
        delta_temperature = self.current_T - past_temperature 
        
        ### Taking the action in account
        delta_temperature_ai = direction * energy_ai  ### delta caused by the AI
        
        ### New temperature with the AI
        self.temperature_ai += delta_temperature_ai + delta_temperature
        
        ### New temperature without AI
        
        self.temperature_noai += delta_temperature
        
            
        ### Check if we lost or ot
        if self.temperature_ai > self.tmax:
            if (self.train ==1):
                self.game_over = 1
            else:
                self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai
                self.temperature_ai = self.optimal_temperature[0]
        if self.temperature_ai < self.tmin:
            if (self.train ==1):
                self.game_over = 1
            else:
                self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1]
                self.temperature_ai = self.optimal_temperature[1]
            
        self.total_energy_ai += energy_ai
        
        ### Scaling the next_state 
        
        users = (self.nb_user - self.min_users) / (self.max_users - self.min_users)
        data_transfer = (self.data_transfer - self.min_rate_data_transfer) / (self.max_rate_data_transfer - self.min_rate_data_transfer)
        temp = (self.temperature_ai - self.tmin) / (self.tmax - self.tmin)
        next_state = np.matrix([users, data_transfer, temp])
        
        return next_state, self.reward, self.game_over
            
    def reset(self, month):
        self.initial_month = month
        self.atmoshperic_temperature = self.monthly_atmospheric_temperature[month]
        self.data_transfer = self.data_transfer_ini
        self.nb_user = self.nb_users_ini
        self.current_T = self.atmoshperic_temperature + 1.25 * (self.nb_user + self.data_transfer)
        self.temperature_ai = self.current_T
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1])/2.0  
        self.game_over = 0
        self.reward = 0
        self.total_energy_ai = 0
        self.total_energy_noai = 0
        
### Method giving us the state, reward, and the gameover
        
    def observation(self):
        users = (self.nb_user - self.min_users) / (self.max_users - self.min_users)
        data_transfer = (self.data_transfer - self.min_rate_data_transfer) / (self.max_rate_data_transfer - self.min_rate_data_transfer)
        temp = (self.temperature_ai - self.tmin) / (self.tmax - self.tmin)
        state = np.matrix([users, data_transfer, temp])
        return state, self.reward, self.game_over
        
            
        
        
        