# -*- coding: utf-8 -*-
# scenes.py
#
"""
Module for generating a scene, or scenery, where an epidemic is
to be simulated by the EpiSiming package.
"""

import typing
import random

import numpy as np
import pandas as pd

import networkx as nx

from collections import namedtuple
from functools import partial

#from episiming import redes, individuais, rede_escolar


class Scene:
    def __init__(self, num_pop: int) -> None:
        self.nome = 'Complete Network'
        self.set_population(num_pop)
        self.set_network()

    def set_population(self, num_pop: int) -> None:
        self.num_pop = num_pop
        self.pop_posicoes = np.random.rand(num_pop, 2)

    def set_network(self) -> None:
        """
        Generates a complete network, connection all individuals together.
        """
        self.network = list(range(self.num_pop))
        
    def set_epidemic_parameters(self, rho_sus, rho_inf, gamma, beta):
        '''
        Set the epidemic parameters.

        Input:
        ------
            rho_sus: float or list of float

            rho_inf: float or list of float
        '''
        if type(rho_sus) == float:
            self.pop_rho_sus = self.num_pop * [rho_sus]
        elif type(rho_sus) == list and len(rho_sus) == self.num_pop:
            self.pop_rho_sus = rho_sus
        else:
            raise AttributeError('Argument rho_sus not of proper type.')

        if type(rho_inf) == float:
            self.pop_rho_inf = self.num_pop * [rho_inf]
        elif type(rho_inf) == list and len(rho_inf) == self.num_pop:
            self.pop_rho_inf = rho_inf   
        else:
            raise AttributeError('Argument rho_inf not of proper type.')
    
        self.gamma = gamma
        self.beta = beta

    def initialize_state(self, pop_state):
        if type(pop_state) == int:
            self.pop_state = self.num_pop*[pop_state]
        elif type(pop_state) == list and len(pop_state) == self.num_pop:
            self.pop_state = pop_state
        else:
            raise AttributeError('Argument pop_state not of proper type.')

    def plot(self, *args, **kargs):
        plt.plot(self.pop_posicoes[:,0], self.pop_posicoes[:,1], *args, **kargs)
            
