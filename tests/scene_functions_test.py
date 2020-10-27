# -*- coding: utf-8 -*-
#
"""
Tests functions in the episiming.scenes.functions module,
which are functions used to generates the scenes.

Test it with `python -m tests.scene_functions_test` from the repo root directory.
"""
import numpy as np
import episiming.scenes.functions as ef


assert(list(ef.get_age_fractions([0, 2], [0.4, 0.6], 4, 'constant'))
       == [0.2, 0.2, 0.3, 0.3]), \
       "Function `get_age_fractions` failed to properly generate age fractions"

assert(len(np.where(ef.symptoms_distribution(15, 1/5)[0] == 0)[0])
       == 3), \
       "Function `symptoms_distribution` failed to properly generate symptomatic and asymptomatic groups"

assert(len(np.where(ef.symptoms_distribution(14, 1/5)[0] == 0)[0])
       == 2), \
       "Function `symptoms_distribution` failed to properly generate symptomatic and asymptomatic groups"


assert(np.abs(np.min(ef.kappa_generator(10, 0.1, 0.1, 3, 2)[0] - np.log(np.log(2)/3))) 
       <= 0.1 and
       np.abs(np.min(ef.kappa_generator(10, 0.1, 0.2, 3, 2)[1] - 2)) 
       <= 0.2 ), \
       "Function `kappa_generator` failed to properly generate random values of eta and delta"