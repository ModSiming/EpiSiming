# -*- coding: utf-8 -*-
# scenes.py
#
"""
Module for the construction of scenes, or scenarios, where an epidemic is
to be simulated by the EpiSiming package via an agent-based model.
"""

import os
import typing
import random
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

import episiming.scenes.functions as funcs

# type hints shotcuts
List = typing.List


class Scene():
    """
    Basic Scene class.

    It should be used for relatively small regions, where the blocks
    have all the same sizes. For regions encompassing several latitudes,
    we need to use variable-length blocks.
    """

    def __init__(self) -> None:
        """
        Instantiates the region.

        This should be replaced in each specific region.
        """
        self.name = 'Template Region'
        self.set_foundation()
        self.set_population()

    def set_foundation(self):
        """
        Defines the population and neighborhood matrices and associated info.

        This should be replaced in each specific region.

        It should define the following attributes:

        Attributes created:
        -------------------
            pop_matrix: numpy.ndarray of type int
                A two-dimensional array, representing a two-dimensional region
                divided in equal-sized rectangular blocks, in which the integer
                value of each coefficient represents the population in the
                corresponding block.

            bl_length_x: float
                The "x" direction length of each block, in any desired unit.

            bl_length_y: float
                The "y" direction length of each block, in any desired unit.

            nbh_matrix: numpy.ndarray
                A two-dimensional array, representing a two-dimensional region
                divided in equal-sized rectangular blocks, in which the
                integer value of each coefficient is the id of a neighborhood.
                Each dimension of this matrix should an integer multiple of
                the corresponding dimension of the population matrix.

            nbh_id_to_name: dict
                A dictionary associating the id of the neighborhood to its
                name. It is assumed that an id of 0 has no neighborhood (e.g.
                reprensenting an inhabitable block), no matter the name.

            nbh_name_to_id: dict
                A dictionary associating the name of the neighborhood to its
                (positive) integer id.

            res_sizes_dens: list
                A list where each index `j` represents the fraction of the
                total number of residences with `j+1` residents.
        """

        self.pop_matrix = np.array([[1, 1], [1, 1]])

        self.bl_length_x = 1
        self.bl_length_y = 1

        self.nbh_matrix = np.array([[1, 2], [3, 4]])

        self.nbh_name_to_id = {'1': 1, '2': 2, '3': 3, '4': 4}
        self.nbh_id_to_name = {1: '1', 2: '2', 3: '3', 4: '4'}

        self.res_sizes_dens = [0.3, 0.4, 0.2, 0.1]

    def set_population(self):
        self.num_pop = self.pop_matrix.sum()

        ydim, xdim = self.pop_matrix.shape

        self.xextent = xdim * self.bl_length_x
        self.yextent = ydim * self.bl_length_y

        xscale = self.nbh_matrix.shape[1] // self.pop_matrix.shape[1]
        yscale = self.nbh_matrix.shape[0] // self.pop_matrix.shape[0]

        self.blsub_length_x = self.bl_length_x / xscale / 10
        self.blsub_length_y = self.bl_length_y / yscale / 10

        self.pop_matrix_fine, _, _ \
            = funcs.interpolate_matrix(self.pop_matrix, self.nbh_matrix)

        self.res_size, self.res_pop, self.pop_res, \
            self.res_bl, self.bl_pop, self.bl_res \
            = funcs.alloc_pop_res_to_blocks(self.pop_matrix,
                                            self.res_sizes_dens)

        self.res_bl_fino, self.res_bl_subbl, self.res_pos \
            = funcs.alloc_res_to_subblocks(self.pop_matrix,
                                           self.pop_matrix_fine,
                                           self.bl_res,
                                           self.bl_length_x,
                                           self.bl_length_y)

        self.res_br = self.nbh_matrix.flatten()[self.res_bl_fino]

        self.pop_pos = funcs.position_pop(self.num_pop,
                                          len(self.res_sizes_dens),
                                          self.res_pop, self.res_pos,
                                          self.blsub_length_x,
                                          self.blsub_length_y)

    def set_susceptibility(self, rho_sus):
        """
        Set the susceptibility factor.

        Input:
        ------
            rho_sus: float or list of float
                Susceptibility factor. Should be positive
        """
        if type(rho_sus) == float:
            self.pop_rho_sus = self.num_pop * [rho_sus]
        elif type(rho_sus) == list and len(rho_sus) == self.num_pop:
            self.pop_rho_sus = rho_sus
        else:
            raise AttributeError(
                'Argument rho_sus should be either a (positive) float or \
a list of (positive) floats')

    def set_infectivity(self, rho_inf):
        """
        Set the infectibility factor.

        Input:
        ------
            rho_inf: float or list of float
                Infectibility factor. Should be positive
        """
        if type(rho_inf) == float:
            self.pop_rho_inf = self.num_pop * [rho_inf]
        elif type(rho_inf) == list and len(rho_inf) == self.num_pop:
            self.pop_rho_inf = rho_inf
        else:
            raise AttributeError(
                'Argument rho_inf should be either a (positive) float or \
a list of (positive) floats')

    def set_kappa(self, delta, eta, eps = 0):
        """
        Set the parameters for the kappa function.

        Input:
        ------
            delta: float
            eta: float
            eps: float

                Parameters of a lognormal distribution function (kappa)
                eps is noise to be added to the parameters for each individual
        """
        self.pop_delta = self.num_pop * np.array([delta]) + np.random.uniform(low=-eps, high=eps, size=(self.num_pop,))
        self.pop_eta = self.num_pop * np.array([delta]) + np.random.uniform(low=-eps, high=eps, size=(self.num_pop,))

    def plot_pop(self, nbh=None, **kargs):
        plt.grid(False)
        plt.xlim(0, self.xextent)
        plt.ylim(0, self.yextent)
        plt.plot(self.pop_pos[:, 0], self.pop_pos[:, 1], 'o', markersize=1,
                 **kargs)

    def plot_res(self, nbh=None, **kargs):
        if nbh:
            if type(nbh) == str and nbh.upper() in self.nbh_name_to_id.keys():
                res_pos_sel \
                    = np.array([pos for pos, br
                                in zip(self.res_pos, self.res_br)
                                if br == self.nbh_name_to_id[nbh.upper()]])
            elif type(nbh) == int and nbh in self.nbh_id_to_name.keys():
                res_pos_sel \
                    = np.array([pos for pos, br
                                in zip(self.res_pos, self.res_br)
                                if br == nbh])
        else:
            res_pos_sel = self.res_pos
        plt.grid(False)
        plt.xlim(0, self.xextent)
        plt.ylim(0, self.yextent)
        plt.plot(res_pos_sel[:, 0], res_pos_sel[:, 1], 'o', markersize=1,
                 **kargs)

    def show_nbh(self, **kargs):
        plt.imshow(self.nbh_matrix, cmap='tab20', interpolation='none',
                   extent=[0, self.xextent, 0, self.yextent], **kargs)


class Random(Scene):
    def __init__(self, num_pop: int, xdim: int, ydim: int) -> None:
        self.name = 'Random'
        self.set_foundation(num_pop, xdim, ydim)
        self.set_population()

    def set_foundation(self, num_pop, xdim, ydim) -> None:
        """
        Defines the population and neighborhood matrices and associated info.

        Creates a region divided in blocks with `xdim` rows and `ydim` columns,
        with the population of size `num_pop` being distributed randomly within
        these blocks. This defines `self.pop.matrix`.

        The sides `self.bl_length_x` and `self.bl_length_y` of the block are
        set to 1.

        A neighborhood matrix `self.nbh_matrix` of dimension xdim * ydim
        are set with neighborhood id going from 1 to xdim * ydim. Each
        neighborhood id `n` is associated with its neighborhood name `"n"`.
        This association gives rise to two dictionaries `self.nbh_name_to_id`
        and `self.nbh_id_to_name`.

        Finally, a synthetic density distribution `self.res_sizes_dens` of
        the fraction of residences per residence size is defined.

        Input:
        ------
            num_pop: int
                population size.

            xdim: int
                number of blocks in the x-direction

            ydim: int
                number of blocks in the y-direction

        Attributes created:
        -------------------

            pop_matrix: numpy.ndarray of type int
                A two-dimensional array, representing a two-dimensional region
                divided in equal-sized rectangular blocks, in which the integer
                value of each coefficient represents the population in the
                corresponding block.

            bl_length_x: float
                The "x" direction length of each block, in any desired unit.

            bl_length_y: float
                The "y" direction length of each block, in any desired unit.

            nbh_matrix: numpy.ndarray
                A two-dimensional array, representing a two-dimensional region
                divided in equal-sized rectangular blocks, in which the
                integer value of each coefficient is the id of a neighborhood.
                Each dimension of this matrix should an integer multiple of
                the corresponding dimension of the population matrix.

            nbh_id_to_name: dict
                A dictionary associating the id of the neighborhood to its
                name. It is assumed that an id of 0 has no neighborhood (e.g.
                reprensenting an inhabitable block), no matter the name.

            nbh_name_to_id: dict
                A dictionary associating the name of the neighborhood to its
                (positive) integer id.

            res_sizes_dens: list
                A list where each index `j` represents the fraction of the
                total number of residences with `j+1` residents.
        """

        self.pop_matrix \
            = np.reshape(
                np.histogram(
                    random.choices(list(range(xdim * ydim)), k=num_pop),
                    bins=np.arange(xdim * ydim + 1))[0],
                (ydim, xdim)
            )

        self.bl_length_x = 1
        self.bl_length_y = 1

        self.nbh_matrix \
            = np.reshape(
                np.arange(1, xdim * ydim + 1).astype(int),
                (ydim, xdim)
            )

        self.nbh_name_to_id \
            = {str(n): n for n in np.arange(1, xdim * ydim + 1)}
        self.nbh_id_to_name \
            = {n: str(n) for n in np.arange(1, xdim * ydim + 1)}

        self.res_sizes_dens = [.25, .3, .2, .15, .05, .03, 0.02]


class RiodeJaneiro(Scene):
    def __init__(self, scale: float = 1) -> None:
        """
        Instantiates the region.

        Defines the `self.name` of the region and calls the methods that
        generate the region and all the parameters.
        """
        self.name = 'Rio de Janeiro'
        self.set_foundation(scale)
        self.set_population()
        self.set_susceptibility()
        self.set_infectivity()
        self.set_kappa()

    def set_foundation(self, scale: float = 1) -> None:
        pop_matrix_file \
            = os.path.join(os.path.dirname(__file__),
                           'dados_rio',
                           'landscan_rio_corrigido.npy')
        self.pop_matrix \
            = (scale * np.load(pop_matrix_file)).astype(int)

        self.bl_length_x = 0.85239
        self.bl_length_y = 0.926

        nbh_matrix_file \
            = os.path.join(os.path.dirname(__file__),
                           'dados_rio',
                           'geoloc_Bairros_MRJ_fino_desloc.npy')
        self.nbh_matrix = np.load(nbh_matrix_file).astype(int)

        with open(os.path.join(os.path.dirname(__file__),
                               'dados_rio',
                               'bairros.yml')) as f:
            nbh = yaml.load(f, Loader=yaml.FullLoader)
            self.nbh_name_to_id = nbh['bairros_id']
            self.nbh_id_to_name = nbh['id_bairros']
            del(nbh)

        self.res_sizes_dens = [.21, .26, .20, .17, .08, .04, .02, 0.02]

        population_pyramid_file \
            = os.path.join(os.path.dirname(__file__),
                           'dados_rio',
                           'piramide_etaria_MRJ.csv')

        population_pyramid = pd.read_csv(population_pyramid_file)
        age_groups = np.array([int(p[0:3])
                               for p in population_pyramid.columns[1:]])

        age_group_fractions \
            = population_pyramid.iloc[0][1:].values \
            / population_pyramid.iloc[0][0]

        age_max = 110

        self.age_fractions \
            = funcs.get_age_fractions(age_groups,
                                      age_group_fractions,
                                      age_max)

    def set_susceptibility(self) -> None:
        rho_sus = np.ones(self.num_pop)

    def set_infectivity(self,
                        rho_forma: float = 0.8,
                        rho_escala: float = 1.25
                        ) -> None:
        """
        Set the infectibility epidemic parameters.

        Input:
        ------
            rho_inf: float or list of float
                Susceptibility factor. Should be positive.
        """
        rho_inf = stats.gamma.rvs(a=rho_forma,
                                  scale=rho_escala,
                                  size=self.num_pop)
    #pegar valores melhores
    def set_kappa(self, 
                  delta: float = .72,
                  eta: float = 1, eps = 0):
        """
        Set the parameters for the kappa function.

        Input:
        ------
            delta: float
            eta: float
            eps: float

                Parameters of a lognormal distribution function (kappa)
                eps is noise to be added to the parameters for each individual
        """
        self.pop_delta = self.num_pop * np.array([delta]) + np.random.uniform(low=-eps, high=eps, size=(self.num_pop,))
        self.pop_eta = self.num_pop * np.array([delta]) + np.random.uniform(low=-eps, high=eps, size=(self.num_pop,))
