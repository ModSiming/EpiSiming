import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

import episiming

rio = episiming.scenes.RiodeJaneiro(1/10)

assert(sum(rio.res_size) == rio.num_pop), \
    "sum(rio.res_size) should be equal to rio.num_pop"

assert(len(rio.pop_pos) == rio.num_pop), \
    "len(rio.pop_pos) should be equal to rio.num_pop"

assert(len(rio.res_pos) == len(rio.res_size)), \
    "len(rio.res_pos) should be equal to len(rio.res_size)"
