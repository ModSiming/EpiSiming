# -*- coding: utf-8 -*-
'''
Joints the jupyter notebooks that make up the collection of lecture notes.
'''

import os
import nbjoint as nbj

os.chdir(os.path.dirname(__file__))

nbj.joint('nbjoint_config.yml')