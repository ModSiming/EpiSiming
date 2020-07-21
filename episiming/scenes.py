# -*- coding: utf-8 -*-
# scenes.py
#
"""
Module for the construction of scenes, or scenarios, where an epidemic is
to be simulated by the EpiSiming.
"""

import os
import typing
import random
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx

from collections import namedtuple
from functools import partial
from scipy.interpolate import interp2d
from scipy import stats

#from episiming import redes, individuais, rede_escolar

# type hints shotcuts
List = typing.List

def interp_matrix(matrix, finer_matrix):
    '''
    Generates an interpolated matrix based on a finer mask matrix.
    
    It uses linear interpolation to obtain a matrix with the same
    dimensions as the mask matrix and with zeros where the mask
    matrix vanishes.  
    '''
    
    if (finer_matrix.shape[0] % matrix.shape[0] 
        + finer_matrix.shape[1] % matrix.shape[1] > 0):
        raise AttributeError(
            'Each dimension of the "finer_matrix" has to be a (positive) ' \
            + 'integer multiple of those of the given "matrix".')
                
    refinement_x = finer_matrix.shape[1] // matrix.shape[1]
    refinement_y = finer_matrix.shape[0] // matrix.shape[0]    
    
    matrix_interpd = np.zeros_like(finer_matrix)
    matrix_fixed = np.copy(matrix)
    not_placed_idx = list() 

    xs = list(range(matrix.shape[1]))
    ys = list(range(matrix.shape[0]))

    xs_fino = np.arange(0, matrix.shape[1], 1/refinement_x)
    ys_fino = np.arange(0, matrix.shape[0], 1/refinement_y)

    if refinement_x * refinement_y == 1:
        matriz_interp = matrix * (np.maximum(np.minimum(finer_matrix,1),0))
    else:
        f = interp2d(xs, ys, matrix, kind='linear')
        matriz_interp \
            = f(xs_fino, ys_fino)*(np.maximum(np.minimum(finer_matrix,1),0))

    for j in xs:
        for i in ys:
            if matrix[i,j]:
                matriz_interp_local \
                    = matriz_interp[i*refinement_y:(i+1)*refinement_y,
                                    j*refinement_x:(j+1)*refinement_x]
                if matriz_interp_local.sum() > 0:
                    distrib = np.floor(matrix[i,j]*matriz_interp_local
                                           / matriz_interp_local.sum()
                                      ).astype('int')
                    remainder = matrix[i,j] - distrib.sum()
                    remainder_placement \
                        = np.random.choice(refinement_x*refinement_y,
                                           remainder,
                                           replace=True,
                                           p=(matriz_interp_local
                                              /matriz_interp_local.sum()
                                             ).flatten()
                                          )

                    for loc in remainder_placement:
                        distrib[loc // refinement_x,
                                loc % refinement_x] += 1

                    matrix_interpd[i*refinement_y:(i+1)*refinement_y,
                                j*refinement_x:(j+1)*refinement_x] \
                        = distrib
                else:
                    not_placed_idx.append([i,j])
                    matrix_fixed[i,j] = 0
                        
    num_pop_displaced = sum([matrix[ij[0], ij[1]] for ij in not_placed_idx])
    if num_pop_displaced > 0:
        distrib_not_place \
            = np.random.choice(finer_matrix.shape[0]*finer_matrix.shape[1],
                               num_pop_displaced,
                               replace=True,
                               p=(matriz_interp/matriz_interp.sum()).flatten()
                              )

        for na in distrib_not_place:
            ii = na // finer_matrix.shape[1]
            jj = na % finer_matrix.shape[1]
            i = ii // refinement_x
            j = jj // refinement_y
            matrix_interpd[ii, jj] += 1
            matrix_fixed[i, j] += 1

    return matrix_interpd, matrix_fixed, num_pop_displaced

def gera_tam_residencias(num_pop, dens_tam_res):
    '''
    Retorna uma "lista de residências", indicando o tamanho de cada uma delas.
    
    A densidade de residências por tamanho de residência `dens_tam_res` 
    é utilizada como densidade de probabilidade para a geração das residências 
    por tamanho.
    '''
    len_tam_res = len(dens_tam_res)
    tam_res = range(1, len_tam_res + 1)
    
    res_tam = list() # tamanho de cada residência
    sobra = num_pop
    while sobra >= 8:
        k = sobra // len_tam_res
        res_tam += random.choices(tam_res, weights=dens_tam_res, k=k)
        sobra = num_pop - sum(res_tam)

    while sobra > 0:
        res_tam += random.choices(tam_res, weights=dens_tam_res)
        sobra = num_pop - sum(res_tam)

    res_tam[-1] = num_pop - sum(res_tam[:-1])
    return res_tam

def associa_pop_residencia(res_tam, res_0=0, ind_0=0):
    '''
    Retorna uma lista com a residência de cada indivíduo
    e uma lista com os indivíduos em cada residência.
    
    Isso é feito a partir da lista do tamanho de cada residência.
    
    A população é associada, por ordem de índice, a cada residência.
    '''
    pop_res = list() # índice da residência de cada indivíduo
    res_pop = list() # índice dos indivíduos em cada residência
    individuo = ind_0
    residencia = res_0
    for k in range(len(res_tam)):
        pop_res += res_tam[k]*[residencia + k]
        res_pop.append(list(range(individuo, individuo + res_tam[k])))
        individuo += res_tam[k]

    return pop_res, res_pop

def distribui_pop_e_res(pop_matrix, dens_tam_res):
    '''
    Distribui as residências e seus residentes pelo reticulado.
    
    Cada coeficiente da matriz populacional `pop_matrix` indica o
    número de indivíduos no bloco correspondente do reticulado associado
    à matriz.
    
    A distribuição das residências e dos seus residentes é feita bloco
    a bloco, através da função `associa_pop_residencia()`.   
    
    Saída:
    ------
    
        res_tam: list of int
            lista indexada pela residência, indicando o tamanho da mesma.
            
        res_pop: list of int
            lista indexada pela residência, indicando a lista de 
            seus residentes.

        pop_res: list of int
            lista indexada pelos indivíduos, indicando o índice
            da sua residência.

        bl_res: list of int
            lista indexada pelo bloco flattened da pop_matrix,
            indicando o índice da sua primeira residência.

        bl_pop: list of int
            lista indexada pelo bloco flattened da pop_matrix,
            indicando o índice do seu primeiro residente.
    '''
    
    ydim, xdim = pop_matrix.shape
    
    res_tam = list()
    res_pop = list() 
    pop_res = list() 
    res_bl = list()

    bl_res = [0]
    bl_pop = [0]

    res_cum = 0
    ind_cum = 0

    for k in range(xdim * ydim):
        num_pop_local = pop_matrix[k // xdim, k % xdim]
        if num_pop_local > 0:
            res_tam_local = gera_tam_residencias(num_pop_local,
                                                 dens_tam_res)
            pop_res_local, res_pop_local \
                = associa_pop_residencia(res_tam_local,
                                         res_cum,
                                         ind_cum)
            res_bl += num_pop_local*[k]
            res_tam += res_tam_local
            pop_res += pop_res_local
            res_pop += res_pop_local
            res_cum += len(res_tam_local)
            ind_cum += num_pop_local
        bl_res.append(res_cum)
        bl_pop.append(ind_cum)
    
    return res_tam, res_pop, pop_res, res_bl, bl_pop, bl_res   

def distrib_res_fina(pop_matrix, matriz_fina, bl_res, bl_length_x, bl_length_y):
    '''
    Distribui as residências pelo reticulado da matriz fina.
    
    Cada coeficiente da matriz populacional `pop_matrix` indica o
    número de indivíduos no bloco correspondente do reticulado associado
    à matriz.
    
    A distribuição das residências e dos seus residências é feita bloco
    a bloco, através da função `associa_pop_residencia()`.    
    '''
    
    if (matriz_fina.shape[0] % pop_matrix.shape[0] 
        + matriz_fina.shape[1] % pop_matrix.shape[1] > 0):
        raise AttributeError(
            'Each dimension of `matrix_fine` should be a multiple of the \n'
            + 'corresponding dimension of `pop_matrix`.')
    
    tx_refinamento_x = matriz_fina.shape[1] // pop_matrix.shape[1]
    tx_refinamento_y = matriz_fina.shape[0] // pop_matrix.shape[0]
    tx_produto = tx_refinamento_x * tx_refinamento_y

    ydim, xdim = pop_matrix.shape

    yextent = ydim * bl_length_y
    
    res_bl_fino = list()
    res_bl_subbl = list()
    
    res_pos = list()

    for l in range(xdim * ydim):
        num_res_local = bl_res[l+1] - bl_res[l]
        if num_res_local > 0:
            i = l // xdim
            j = l % xdim
            matriz_fina_local \
                    = matriz_fina[i*tx_refinamento_y:(i+1)*tx_refinamento_y,
                                  j*tx_refinamento_x:(j+1)*tx_refinamento_x]

            if matriz_fina_local.sum() > 0:
                distrib \
                    = np.random.choice(tx_produto,
                                       num_res_local,
                                       replace=True,
                                       p=(matriz_fina_local
                                          /matriz_fina_local.sum()
                                         ).flatten()
                                      )
                num_res_loc_fino, _ = np.histogram(
                    distrib,
                    bins=np.arange(tx_produto+1)
                )
                for l_loc in range(tx_produto):
                    l_fino = ( i * tx_refinamento_y + l_loc // tx_refinamento_x) \
                                * xdim * tx_refinamento_x \
                                + j*tx_refinamento_x + l_loc % tx_refinamento_x
                    num_res_l_loc = num_res_loc_fino[l_loc]
                    res_bl_subbl += num_res_l_loc*[l_loc]
                    res_bl_fino += num_res_l_loc*[l_fino]
                    
                    if num_res_l_loc > 0:
                        sorteio = random.choices(list(range(tx_produto)), k=num_res_l_loc)
                        x_0 = (l_fino % (tx_refinamento_x * xdim)) * bl_length_x / tx_refinamento_x
                        y_0 = (l_fino // (tx_refinamento_x * xdim)) * bl_length_y / tx_refinamento_y
                        res_pos_loc = [ (x_0 + (k % tx_refinamento_x + 1/2) / tx_refinamento_x / tx_refinamento_x,
                                         yextent - y_0 - (k // tx_refinamento_x + 1/2) / tx_refinamento_x / tx_refinamento_y
                                        )
                                       for k in sorteio
                                      ]
                        res_pos += res_pos_loc
                
            else:
                raise AttributeError('Populated block from population matrix'
                    + ' without designated neighboorhood'
                )
    
    res_pos = np.array(res_pos)
    return res_bl_fino, res_bl_subbl, res_pos

def posiciona_pop(num_pop, res_tam_max, res_pop, res_pos,
                  micro_escala_x, micro_escala_y):
    template = [np.array([(0,0)])]

    for m in range(1, res_tam_max):
        template.append(
            np.array([(np.cos(i*2*np.pi/(m+1))/2 * micro_escala_x,
                       np.sin(i*2*np.pi/(m+1))/2 * micro_escala_y)
                      for i in range(m+1)
                     ]
                    )
        )
    pop_pos = np.zeros([num_pop,2])
    for r, pos in zip(res_pop, res_pos):
        pop_pos[r] = np.array(pos) + template[len(r)-1]      

    return pop_pos

def get_age_fractions(age_groups: List[int],
                      age_group_fractions: List[int],
                      age_max: int =100,
                      interp: str ='linear'):
    '''
    Interpolates the population pyramid.
    
    Population pyramids are usually available in group ages, gathering
    a number of years in each group. This function reads the group ages,
    their population fractions and a maximum desired age and interpolates 
    the data to output a pyramid with data for every year.
    
    The interpolation can be either 'constant' by parts or 'linear' by parts.
    
    Input:
    ------
        age_groups: list of int
            Each element of the list indicates the first age year in the group.
            
        age_group_fractions: list of float
            Each element i of the list is expected to be a float 
            between 0 and 1, indicating the faction of the population
            in the age group from age_groups[i] to age_groups[i+1], if
            i is not the last index, or from age_groups[i] to age_max, 
            if i is the last index
            
        age_max: int
            The maximum age for the output pyramid
            
        interp: str
            The type of interpolation, which can be either piecewise 
            'constant' (default) or piecewise 'linear'.
    
    Output:
    -------
        age_fractions: list of float
            Each element age_fractions[i] of the list indicates the
            fraction of the population at age i.
    '''
    
    # interpola/extrapola pirâmide populacional
    age_fractions = list()
    
    if interp == 'linear':
        for j in range(len(age_groups)-1):
            age_fractions += (age_groups[j+1] - age_groups[j]) \
                * [age_group_fractions[j]/(age_groups[j+1]-age_groups[j])]
        age_fractions += (age_max - age_groups[-1]) \
            * [age_group_fractions[-1]/(age_max-age_groups[-1])]
    elif interp == 'constant':
        for j in range(len(age_groups)-1):
            age_fractions += (age_groups[j+1] - age_groups[j]) \
                * [age_group_fractions[j]/(age_groups[j+1]-age_groups[j])]
        age_fractions += (age_max - age_groups[-1]) \
            * [age_group_fractions[-1]/(age_max-age_groups[-1])]
    else:
        raise ValueError("Argument 'interp' should be either 'linear' or 'constant'.")
    
#    age_fractions = np.array(age_fractions)

    return age_fractions

class Scene():
    """
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
        self.pop_matrix = np.array([[1,1],[1,1]])

        self.bl_length_x = 1
        self.bl_length_y = 1

        self.nbh_matrix = np.array([[1,2],[3,4]])

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
            = interp_matrix(self.pop_matrix, self.nbh_matrix)

        self.res_size, self.res_pop, self.pop_res, \
            self.res_bl, self.bl_pop, self.bl_res \
            = distribui_pop_e_res(self.pop_matrix, self.res_sizes_dens)

        self.res_bl_fino, self.res_bl_subbl, self.res_pos \
            =  distrib_res_fina(self.pop_matrix, self.pop_matrix_fine,
                                self.bl_res, self.bl_length_x,
                                self.bl_length_y)

        self.res_br = self.nbh_matrix.flatten()[self.res_bl_fino]

        self.pop_pos = posiciona_pop(self.num_pop, len(self.res_sizes_dens),
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

    def set_infectibility(self, rho_inf):
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

    def plot_pop(self, nbh=None, **kargs):                  
        plt.grid(False)
        plt.xlim(0, self.xextent)
        plt.ylim(0, self.yextent)
        plt.plot(self.pop_pos[:,0], self.pop_pos[:,1], 'o', markersize=1, 
                 **kargs)

    def plot_res(self, nbh=None, **kargs):
        if nbh and (nbh.upper() in self.nbh_name_to_id.keys() or
                     nbh in self.nbh_id_to_name.keys()):
            if type(nbh) == str:
                res_pos_sel \
                    = np.array([self.res_pos[k] 
                                for k in range(len(self.res_pos))
                                if self.res_br[k] \
                                    == self.nbh_name_to_id[nbh.upper()]])
            else:
                res_pos_sel \
                    = np.array([self.res_pos[k] 
                                for k in range(len(self.res_pos))
                                if self.res_br[k] \
                                    == nbh])
        else:
            res_pos_sel = self.res_pos
        plt.grid(False)
        plt.xlim(0, self.xextent)
        plt.ylim(0, self.yextent)
        plt.plot(res_pos_sel[:,0], res_pos_sel[:,1], 'o', markersize=1, 
                 **kargs)

    def show_nbh(self, **kargs):
        plt.imshow(self.nbh_matrix, cmap='tab20', interpolation='none',
           extent=[0,self.xextent,0,self.yextent], **kargs)

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
                    bins = np.arange(xdim * ydim + 1))[0],
                (ydim, xdim)
            )

        self.bl_length_x = 1
        self.bl_length_y = 1

        self.nbh_matrix \
            = np.reshape(
                np.arange( 1, xdim * ydim + 1).astype(int),
                (ydim, xdim)
            )

        self.nbh_name_to_id = {str(n): n for n in np.arange(1, xdim * ydim + 1)}
        self.nbh_id_to_name = {n: str(n) for n in np.arange(1, xdim * ydim + 1)}

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
        self.set_infectibility()

    def set_foundation(self, scale: float = 1) -> None:
        pop_matrix_file \
            = os.path.join('..', 'input', 'dados_rio',
                           'landscan_rio_corrigido.npy')
        self.pop_matrix \
            = (scale * np.load(pop_matrix_file)).astype(int)

        self.bl_length_x = 0.85239
        self.bl_length_y = 0.926

        nbh_matrix_file \
            = os.path.join('..', 'input', 'dados_rio',
                           'geoloc_Bairros_MRJ_fino_desloc.npy')
        self.nbh_matrix = np.load(nbh_matrix_file).astype(int)

        with open(os.path.join('..', 'input', 'dados_rio', 
                               'bairros.yml')) as f:
            nbh = yaml.load(f, Loader=yaml.FullLoader)
            self.nbh_name_to_id = nbh['bairros_id']
            self.nbh_id_to_name = nbh['id_bairros']
            del(nbh)

        self.res_sizes_dens = [.21, .26, .20, .17, .08, .04, .02, 0.02]

        population_pyramid_file \
            = os.path.join('..',
                        'input',
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
            = get_age_fractions(age_groups,
                                age_group_fractions,
                                age_max
                            )

    def set_susceptibility(self) -> None:
        rho_sus = np.ones(self.num_pop)

    def set_infectibility(self,
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
        rho = stats.gamma.rvs(a=rho_forma,
                              scale=rho_escala,
                              size=self.num_pop)

        


        
