# -*- coding: utf-8 -*-
# scenes.py
#
"""
Module with functions to aid the construction of the scenes.
"""

import typing
import random

import numpy as np

from scipy.interpolate import interp2d
from scipy.stats import lognorm, weibull_min

# type hints shotcuts
List = typing.List


def interpolate_matrix(matrix, finer_matrix):
    """
    Generates an interpolated matrix based on a finer mask matrix.

    It uses linear interpolation to obtain a matrix with the same
    dimensions as the mask matrix and with zeros where the mask
    matrix vanishes.
    """

    if (finer_matrix.shape[0] % matrix.shape[0]
            + finer_matrix.shape[1] % matrix.shape[1] > 0):
        raise AttributeError(
            'Each dimension of the "finer_matrix" has to be an integer '
            + 'multiple of those of the given "matrix".')

    refinement_x = finer_matrix.shape[1] // matrix.shape[1]
    refinement_y = finer_matrix.shape[0] // matrix.shape[0]

    matrix_interpd = np.zeros_like(finer_matrix)
    matrix_fixed = np.copy(matrix)
    not_placed_idx = list()

    xs = list(range(matrix.shape[1]))
    ys = list(range(matrix.shape[0]))

    xs_fino = np.arange(0, matrix.shape[1], 1 / refinement_x)
    ys_fino = np.arange(0, matrix.shape[0], 1 / refinement_y)

    if refinement_x * refinement_y == 1:
        matriz_interp = matrix * (np.maximum(np.minimum(finer_matrix, 1), 0))
    else:
        f = interp2d(xs, ys, matrix, kind='linear')
        matriz_interp \
            = f(xs_fino, ys_fino) * (np.maximum(np.minimum(finer_matrix, 1), 0))

    for j in xs:
        for i in ys:
            if matrix[i, j]:
                matriz_interp_local \
                    = matriz_interp[i * refinement_y:(i + 1) * refinement_y,
                      j * refinement_x:(j + 1) * refinement_x]
                if matriz_interp_local.sum() > 0:
                    distrib = np.floor(matrix[i, j] * matriz_interp_local
                                       / matriz_interp_local.sum()
                                       ).astype('int')
                    remainder = matrix[i, j] - distrib.sum()
                    remainder_placement \
                        = np.random.choice(refinement_x * refinement_y,
                                           remainder,
                                           replace=True,
                                           p=(matriz_interp_local
                                              / matriz_interp_local.sum()
                                              ).flatten()
                                           )

                    for loc in remainder_placement:
                        distrib[loc // refinement_x,
                                loc % refinement_x] += 1

                    matrix_interpd[i * refinement_y:(i + 1) * refinement_y,
                    j * refinement_x:(j + 1) * refinement_x] \
                        = distrib
                else:
                    not_placed_idx.append([i, j])
                    matrix_fixed[i, j] = 0

    num_pop_displaced = sum([matrix[ij[0], ij[1]] for ij in not_placed_idx])
    if num_pop_displaced > 0:
        distrib_not_place \
            = np.random.choice(finer_matrix.shape[0] * finer_matrix.shape[1],
                               num_pop_displaced,
                               replace=True,
                               p=(matriz_interp / matriz_interp.sum()).flatten()
                               )

        for na in distrib_not_place:
            ii = na // finer_matrix.shape[1]
            jj = na % finer_matrix.shape[1]
            i = ii // refinement_x
            j = jj // refinement_y
            matrix_interpd[ii, jj] += 1
            matrix_fixed[i, j] += 1

    return matrix_interpd, matrix_fixed, num_pop_displaced


def gen_res_size(num_pop, dens_tam_res):
    """
    Generates a list with the size of each residence.

    A densidade de residências por tamanho de residência `dens_tam_res`
    é utilizada como densidade de probabilidade para a geração das residências
    por tamanho.
    """
    len_tam_res = len(dens_tam_res)
    tam_res = range(1, len_tam_res + 1)

    res_tam = list()  # tamanho de cada residência
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


def link_pop_and_res(res_tam, res_0=0, ind_0=0):
    """
    Returns two lists linking residences to its individuals and vice-versa.

    Retorna uma lista com a residência de cada indivíduo
    e uma lista com os indivíduos em cada residência.

    Isso é feito a partir da lista do tamanho de cada residência.

    A população é associada, por ordem de índice, a cada residência.
    """
    pop_res = list()  # índice da residência de cada indivíduo
    res_pop = list()  # índice dos indivíduos em cada residência

    res = res_0
    ind = ind_0
    for size in res_tam:
        pop_res += size * [res]
        res_pop.append(list(range(ind, ind + size)))
        res += 1
        ind += size

    return pop_res, res_pop


def alloc_pop_res_to_blocks(pop_matrix, dens_tam_res):
    """
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
    """

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
            res_tam_local = gen_res_size(num_pop_local, dens_tam_res)
            pop_res_local, res_pop_local \
                = link_pop_and_res(res_tam_local, res_cum, ind_cum)
            res_bl += num_pop_local * [k]
            res_tam += res_tam_local
            pop_res += pop_res_local
            res_pop += res_pop_local
            res_cum += len(res_tam_local)
            ind_cum += num_pop_local
        bl_res.append(res_cum)
        bl_pop.append(ind_cum)

    return res_tam, res_pop, pop_res, res_bl, bl_pop, bl_res


def alloc_res_to_subblocks(pop_matrix, matriz_fina, bl_res,
                           bl_length_x, bl_length_y):
    """
    Distribui as residências pelo reticulado da matriz fina.

    Cada coeficiente da matriz populacional `pop_matrix` indica o
    número de indivíduos no bloco correspondente do reticulado associado
    à matriz.

    A distribuição das residências e dos seus residências é feita bloco
    a bloco, através da função `associa_pop_residencia()`.
    """

    if (matriz_fina.shape[0] % pop_matrix.shape[0]
            + matriz_fina.shape[1] % pop_matrix.shape[1] > 0):
        raise AttributeError(
            'Each dimension of `matrix_fine` should be an integer multiple\n'
            + 'of the corresponding dimension of `pop_matrix`.')

    tx_refinamento_x = matriz_fina.shape[1] // pop_matrix.shape[1]
    tx_refinamento_y = matriz_fina.shape[0] // pop_matrix.shape[0]
    tx_produto = tx_refinamento_x * tx_refinamento_y

    ydim, xdim = pop_matrix.shape

    yextent = ydim * bl_length_y

    res_bl_fino = list()
    res_bl_subbl = list()

    res_pos = list()

    for l in range(xdim * ydim):
        num_res_local = bl_res[l + 1] - bl_res[l]
        if num_res_local > 0:
            i = l // xdim
            j = l % xdim
            matriz_fina_local \
                = matriz_fina[i * tx_refinamento_y:(i + 1) * tx_refinamento_y,
                  j * tx_refinamento_x:(j + 1) * tx_refinamento_x]

            if matriz_fina_local.sum() > 0:
                distrib \
                    = np.random.choice(tx_produto,
                                       num_res_local,
                                       replace=True,
                                       p=(matriz_fina_local
                                          / matriz_fina_local.sum()
                                          ).flatten()
                                       )
                num_res_loc_fino, _ = np.histogram(
                    distrib,
                    bins=np.arange(tx_produto + 1)
                )
                for l_loc in range(tx_produto):
                    l_fino = (i * tx_refinamento_y
                              + l_loc // tx_refinamento_x) \
                             * xdim * tx_refinamento_x \
                             + j * tx_refinamento_x + l_loc % tx_refinamento_x
                    num_res_l_loc = num_res_loc_fino[l_loc]
                    res_bl_subbl += num_res_l_loc * [l_loc]
                    res_bl_fino += num_res_l_loc * [l_fino]

                    if num_res_l_loc > 0:
                        sorteio = random.choices(list(range(tx_produto)),
                                                 k=num_res_l_loc)
                        x_0 = (l_fino % (tx_refinamento_x * xdim)) \
                              * bl_length_x / tx_refinamento_x
                        y_0 = (l_fino // (tx_refinamento_x * xdim)) \
                              * bl_length_y / tx_refinamento_y
                        res_pos_loc = [(x_0 + (k % tx_refinamento_x + 1 / 2)
                                        / tx_refinamento_x / tx_refinamento_x,
                                        yextent - y_0
                                        - (k // tx_refinamento_x + 1 / 2)
                                        / tx_refinamento_x / tx_refinamento_y
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


def position_pop(num_pop, res_tam_max, res_pop, res_pos,
                 micro_escala_x, micro_escala_y):
    template = [np.array([(0, 0)])]

    for m in range(1, res_tam_max):
        template.append(
            np.array([(np.cos(i * 2 * np.pi / (m + 1)) / 2 * micro_escala_x,
                       np.sin(i * 2 * np.pi / (m + 1)) / 2 * micro_escala_y)
                      for i in range(m + 1)
                      ]
                     )
        )
    pop_pos = np.zeros([num_pop, 2])
    for r, pos in zip(res_pop, res_pos):
        pop_pos[r] = np.array(pos) + template[len(r) - 1]

    return pop_pos


def get_age_fractions(age_groups: List[int],
                      age_group_fractions: List[float],
                      age_max: int = None,
                      interp: str = 'constant'):
    """
    Interpolates the population pyramid.

    Population pyramids are usually available in group ages, gathering
    a number of years in each group. This function reads the group ages,
    their population fractions and a maximum desired age and interpolates
    the data to output a pyramid with data for every year.

    The interpolation implemented so far is 'constant' by parts.

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
            The maximum age for the output age pyramid. If not given,
            the maximum age is taken to be the maximum between 100
            and the last age in the list plus the difference between
            the last age and the age before the last one.

        interp: str
            The type of interpolation. Currently, only 'constant' is
            implemented, which is taken as default value, leading
            to a constant by parts interpolation.

    Output:
    -------
        age_fractions: list of float
            Each element age_fractions[i] of the list indicates the
            fraction of the population at age i.
    """

    # interpola/extrapola pirâmide populacional
    age_fractions = list()

    if not age_max:
        age_max = max(100, 2 * age_groups[-1] - age_groups[-2])

    if interp == 'constant':
        for a1, a, af in zip(age_groups[1:], age_groups[:-1],
                             age_group_fractions[:-1]):
            # copies a1-a times the constant value af / (a1 - a)
            age_fractions += (a1 - a) * [af / (a1 - a)]

        # analogous
        age_fractions += (age_max - age_groups[-1]) \
            * [age_group_fractions[-1] / (age_max - age_groups[-1])]

    else:
        raise ValueError("Only 'constant' is accepted for the 'interp' \
argument")

    return age_fractions

def set_subnot(dic_cases, subnot):
    rng = dic_cases.keys()
    n_cases = [x * subnot for x in dic_cases.values()]
    return dict(zip(rng, n_cases))

def rescale_cases(dic_cases, scale):
    if scale == 1:
        rescaled_cases = dic_cases
    else:
        cases = np.array(list(dic_cases.values()))
        total_rescaled_cases = np.rint(np.sum(cases) * scale)
        weights = (cases / np.sum(cases))
        dummy = np.random.choice(list(dic_cases.keys()), p=weights, size=int(total_rescaled_cases))
        rescaled_cases = dict(zip(dic_cases.keys(), np.zeros(len(dic_cases.keys()))))
        for i in dummy:
            rescaled_cases[i] += 1
    return rescaled_cases

def start_case_distribution(scale, dic_cases, mtrx_locations, res_pos, res_pop, res_br, pop_pos, c = 1, d = 2.5):
    """
    
    
    
    Input:
    ------
    scale:
    
    dic_cases:
    
    mtrx_locations:
    
    res_pos:
    
    res_pop:
    
    res_br:
    
    pop_pos:
    
    c: float
        shape parameter of weibull distribution
   
   d: float
       scale parameter of weibull distribution
       
   Output:
   -------
    cases: dict
        dictionary of index of the person infected and time since infection
    
    """
    locations_block = np.array(mtrx_locations[int(x[1]), int(x[0])] for x in res_pos)
    rescaled_cases = rescale_cases(dic_cases, scale)

    beta = .6

    cases_infect = []
    ids_location = list(dic_cases.keys()) # ids from the locations (bairros, AP, RP, RA)
    rng = np.arange(len(res_pos))
    for i in ids_location:
        # res_br array, res_location array booleano
        res_location = res_br == i # Get residences from current location 
        res_location_index = rng[res_location] # select
        n_cases = rescaled_cases[i] # Number of cases from current location
        if (n_cases > 0) & (len(res_location_index) > 0):
            j = 0
            dist = 0 # ?
            res_permt = np.random.permutation(res_location_index) # permuta as casas 
            while (n_cases > 0 and j < len(res_permt)):
                res_cases = res_permt[j] # primeira casa após a permutação
                first_case = np.random.choice(res_pop[res_cases]) # escolhe uma pessoa na casa(pelo indice na populacao) 
                first_case_index = res_pop[res_cases].index(first_case) # pega o indice da pessoa na casa
                n_res = len(res_pop[res_cases]) # numero de pessoas na casa
                make_infection = np.random.rand(n_res) # gera um vetor aleatorio (entre 0 e 1) do tamanho do numero de pessoas na casa
                make_infection[first_case_index] = 1 # o primeiro infectado tem que estar infectado
                rng_res = np.arange(n_res) # copia da residencia
                infectious_index = rng_res[make_infection > beta] # Indice relativo a casa dos infectados
                infecteds = np.array(res_pop[res_cases])[infectious_index] # Indice geral dos infectados de dada casa
                if n_cases - len(infecteds) < 0: # se passou, jogar casos fora
                    infecteds = infecteds[:int(n_cases)] # joga os últimos casos fora
                n_cases -= len(infecteds) 
                dist += len(infecteds) # ?
                cases_infect.append(infecteds)
                j += 1
                

    ys = np.floor(d*scipy.stats.weibull_min.rvs(c, size = len(np.hstack(cases_infect))))
    cases = {k:ys[i] for (i,k) in enumerate(np.hstack(cases_infect))} # k = indice da pessoa : tempo de infecção (já com a weibull)
    return cases

def kappa_generator(num_pop, eps1, ep2, eta = np.sqrt(-2*np.log(np.log(2))), gamma = 1/2.6, loc = 0.2, factor = 3.5): 
    """
    Generates the kappa function.

    Each individual is assigned to a lognorm pdf
    such that the parameters are given by set values
    delta and eta plus some noise eps.

    Input:
    ------
        
        res_pop: list
            Nested list of individuals by residence

        eps1: float
            maximum value of noise to be added to delta 
            
        eps2: float
            maximum value of noise to be added to eta 
            
        eta: float
            scale parameter of the lognorm pdf
        
        gamma: float
            the s parameter of the lognorm pdf is given by log(2)/gamma
        
        loc: float
            loc parameter of the lognorm pdf

        factor: float
            factor multiplying the lognorm pdf

    Output:
    -------
        noises: list
            list of deltas and etas for each person in population
    """
    
    delta = np.log(np.log(2)/gamma)
    pop = np.arange(num_pop)
    noise_delta = delta + eps1 * np.random.rand(pop)
    noise_eta = eta + eps2 * np.random.rand(pop)
    return [noise_delta, noise_eta]

def weighted(l):
    '''
    Make a distribution out of a list or array.
    
    Given any data array or list, this function returns its distribution, or weights which sum to 1
    
    Input:
    ------
        l: list of float/int
            List or Array of data to be made into a distribution
    
    Output:
    -------
        weighted: array of float/int
            The same data as weighted array
    '''
    return np.array(l)/sum(np.array(l))

def sample_from(weights, x):
    '''
    Draw a random sample from desired distribution using the uniform distribution,
    this function mimics np.random.choice, but it's faster if we need to call it a lot of times
    when it's not possible to vectorize.
    
    Input:
    ------
        weights: array of float
            The weights of the distribution
            
        x: float
            Sample previously drawn from uniform distribution, using np.random.rand()
    
    Output:
    -------
        weighted: int
            Index of drawn sample from weights
    '''
    return (weights.cumsum() > x).argmax()

def gen_pop_age(res, res_size, weights, age_groups, n_pop, adult_age = 19, two_res_prob = 0.9):
    """
    Generates the age for the entire population, following a distribution

    Each residence is guaranteed to have at least one adult individual.
    For residences with two individuals, there's a slight chance that the second individual is not an adult
    The age which separates an adult from a non-adult is given by a parameter, and a sigmoid function is built upon this age.
    
    Input:
    ------
        
        res_pop: list
            Nested list of residences with it's individuals
            
        res_size: list
            List with sizes of residences

        weights: array of floats
            Age distribution/weights for the location, this array needs to sum to 1

        age_groups: Array of int
            Group of ages that the model will gather individuals
        
        n_pop: int
            Total number of the population
        
        adult_age: float
            The age which the model should separate an adult from a non-adult
        
        two_res_prob: float
            Probability of a second individual on a two size residence is an adult

    Output:
    -------
        ages: Array of int
            An array with the size of the population,
            each entry representing the age of the individual corresponding to the index on the array
    """
    n_ages = len(weights)
    ages = -1 * np.ones(n_pop, int)
    max_people = np.ceil(n_pop * weights / sum(weights))
    aux = np.random.rand(n_pop)
    age_range = [(np.tanh(i * (age_groups - adult_age)) + 1) / 2 for i in [2/3, -2/3]]
    rng_ages = range(n_ages)
    
    ages_res_adults = np.random.choice(rng_ages, p = weighted(max_people*age_range[0]), size = len(res))
    ages_res_two = np.random.choice(rng_ages, p = weighted(max_people * (age_range[1] if np.random.rand() > two_res_prob else age_range[0])), size = np.count_nonzero(res_size == 2))
    k_adults = 0
    k_two_size = 0
    for i in res:
        if len(i) > 0:
            ages[i[0]] = ages_res_adults[k_adults]
            max_people[ages[i[0]]] -= 1
            k_adults += 1
            if len(i) == 2:
                ages[i[1]] = ages_res_two[k_two_size]
                max_people[ages[i[1]]] -= 1
                k_two_size +=1
     
    ages_res = np.random.choice(range(n_ages), p = weighted(max_people), size = np.count_nonzero(ages == -1))
    j = 0    
    for i in res:
        if len(i) > 2:
            ages[i[1:]] = ages_res[j:(j+len(i) -1)]
            max_people[ages[i[1:]]] -= 1
            j += len(i) -1
    return ages

