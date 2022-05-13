#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:00:58 2022

@author: ubuntu
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10)
y = np.arange(-10, 10)

# show image
fig, ax=plt.subplots()
plt.grid(visible = True)
plt.show()
#ax.imshow(img)

# select point
yroi = plt.ginput(0,0)
np.save('room1_finer.npy', np.array(yroi))
#np.save('obstacle_bar8.npy', np.array(yroi))

