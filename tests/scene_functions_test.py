# -*- coding: utf-8 -*-
#
"""
Tests functions in the episiming.scenes.functions module,
which are functions used to generates the scenes.

Test it with `python -m tests.functions_test` from the repo root directory.
"""

import episiming.scenes.functions as ef

assert(ef.get_age_fractions([0, 2], [0.5, 0.5], 4)
       == [0.25, 0.25, 0.25, 0.25]), \
       "blah"
