
################################################################
# Laplace Neural Operator - https://arxiv.org/abs/2303.10528
# Code is provided here: https://github.com/qianyingcao/Laplace-Neural-Operator
# But ideally should be rewritten using the FNO as the base framework. 
################################################################


import numpy as np 
import torch 
import torch.nn as nn 
import torch.functional as F 

import operator
from functools import reduce
from functools import partial
from collections import OrderedDict
