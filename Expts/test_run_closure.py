

# %% 

configuration = {"Folder": 'Test',
                 "Issue": 'run_closure'
                 }

import os
from simvue import Run
run = Run(mode='online')
run.init(folder="/Tests", tags=['Debug'], metadata=configuration)

import torch 

aa = torch.rand(2,3)
bb = torch.rand(2,3)

cc = torch.matmul(aa, bb)

# %% 