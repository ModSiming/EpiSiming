# -*- coding: utf-8 -*-
#
"""
Tests the classes that generates the scenes.

Currently, only the `Rio de Janeiro` class is implemented.

Test it with `python -m tests.scenes_test` from the repo root directory.
"""

import episiming

rio = episiming.scenes.RiodeJaneiro(1/100)

assert(sum(rio.res_size) == rio.num_pop), \
    "sum(rio.res_size) should be equal to rio.num_pop"

assert(len(rio.pop_pos) == rio.num_pop), \
    "len(rio.pop_pos) should be equal to rio.num_pop"

assert(len(rio.res_pos) == len(rio.res_size)), \
    "len(rio.res_pos) should be equal to len(rio.res_size)"
