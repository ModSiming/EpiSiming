# -*- coding: utf-8 -*-
#
"""
Tests functions in the episiming.scenes.functions module,
which are functions used to generates the scenes.

Test it with `python -m tests.functions_test` from the repo root directory.
"""

import episiming.scenes.functions as ef

assert(list(ef.get_age_fractions([0, 2], [0.4, 0.6], 4, 'constant'))
       == [0.2, 0.2, 0.3, 0.3]), \
       "Function `get_age_fractions` failed to properly generate age fractions"
