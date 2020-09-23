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
                      interp: str = 'linear'):
    """
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
    """

    # interpola/extrapola pirâmide populacional
    age_fractions = list()

    if not age_max:
        age_max = max(100, 2 * age_groups[-1] - age_groups[-2])

    if interp == 'constant' or interp == 'linear':
        for a1, a, af in zip(age_groups[1:], age_groups[:-1],
                             age_group_fractions[:-1]):
            age_fractions += (a1 - a) * [af / (a1 - a)]

        age_fractions += (age_max - age_groups[-1]) \
                         * [age_group_fractions[-1] / (age_max - age_groups[-1])]
    else:
        raise ValueError("Argument 'interp' should be either 'linear'"
                         + "or 'constant'.")

    #    age_fractions = np.array(age_fractions)

    return age_fractions

    def set_subnot(dic_cases, subnot):
        rng = dic_casos.keys()
        n_cases = [x * subnot for x in dic_cases.values()]
        return dict(zip(rng, n_cases))

    def rescale_cases(scale, dic_cases):
        if scale == 1:
            rescaled_cases = dic_cases
        else:
            cases = np.array(list(dic_cases.values()))
            total_rescaled_cases = np.rint(np.sum(cases) / scale)
            weights = (cases / np.sum(cases))
            dummy = np.random.choice(list(dic_cases.keys()), p=weights, size=int(total_rescaled_cases))
            rescaled_cases = dict(zip(dic_cases.keys(), np.zeros(len(dic_cases.keys()))))
            for i in dummy:
                rescaled_cases[i] += 1
        return rescaled_cases

    def start_case_distribution(scale, dic_cases, mtrx_locations, res_pos, pop_pos):

        return 0

    def distribuicao_inicial_casos(tx_reducao, dic_casos, mtrx_bairros, res_posicoes, res_individuos):
        mtrx_bairros = corrige_mtrx(mtrx_bairros, 1)
        pos_res_blocos = np.round(np.array(res_posicoes) * [9.9, 9.8])
        posicao_bairro = np.array([mtrx_bairros[int(x[1]), int(x[0])] for x in pos_res_blocos])
        dic_casos_reduzida = reducao_casos(tx_reducao, dic_casos)

        beta = .6
        casos = []
        ids_bairros = np.arange(1, 164)
        rng = np.arange(len(pos_res_blocos))
        for i in ids_bairros:
            res_no_bairro = posicao_bairro == i
            indices_no_bairro = rng[res_no_bairro]
            qt_casos = dic_casos_reduzida[i]
            if (qt_casos > 0) & (len(indices_no_bairro) > 0):
                j = 0
                dist = 0
                permt_residencias = np.random.permutation(indices_no_bairro)
                while (qt_casos > 0 and j < len(permt_residencias)):
                    res_caso = permt_residencias[j]
                    caso_primario = np.random.choice(res_individuos[res_caso])
                    indice_primario = res_individuos[res_caso].index(caso_primario)
                    qt_casa = len(res_individuos[res_caso])
                    infecta = np.random.rand(qt_casa)
                    infecta[indice_primario] = 1
                    rng_res = np.arange(qt_casa)
                    indice_infectados = rng_res[infecta > beta]
                    infectados = np.array(res_individuos[res_caso])[indice_infectados]
                    if qt_casos - len(infectados) < 0:
                        infectados = infectados[:int(qt_casos)]
                    qt_casos -= len(infectados)
                    dist += len(infectados)
                    casos.append(infectados)
                    j += 1
        return np.hstack(casos)

def weights(list):
    return list / sum(list)

def gen_age(residencias, age_fractions, age_range):
    n = len(residencias)
    n_ages = len(age_fractions)
    ages = -1 * np.ones(n, int)
    max_people = np.ceil(n * age_fractions / sum(age_fractions))

    aux = [np.arange(n, dtype=int)[residencias == i] for i in range(max(residencias)+1)]

    for i in aux:
        ages[i[0]] = np.random.choice(range(n_ages), p=weights(max_people * age_range[0]))
        max_people[ages[i[0]]] -= 1
        if len(i) == 2:
            ages[i[2]] = np.random.choice(range(n_ages), p=weights(max_people * (age_range[1] if np.random.rand() > 0.9 else age_range[0])))
            max_people[ages[i[2]]] -= 1

    for i in aux:
        if len(i) > 2:
            for j in range(1, len(i)):
                ages[i[j]] = np.random.choice(range(n_ages), p=weights(max_people))
                max_people[ages[i[j]]] -= 1
    return ages