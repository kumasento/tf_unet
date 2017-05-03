#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

# import matplotlib.pyplot as plt
# import matplotlib
import numpy as np
import glob
# plt.rcParams['image.cmap'] = 'gist_earth'

from radio_util import DataProvider
from tf_unet import unet
files = glob.glob('../bgs_example_data/seek_cache/*')

# read data
data_provider = DataProvider(600, files)

# setting up the unet
net = unet.Unet(channels=data_provider.channels,
                n_class=data_provider.n_class,
                layers=3,
                features_root=64,
                cost_kwargs=dict(regularizer=0.001),
               )

# training the network
trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(data_provider, "./unet_trained_bgs_example_data",
                     training_iters=32,
                     epochs=1,
                     dropout=0.5,
                     display_step=2)

# running the prediction on the trained unet
# data_provider = DataProvider(10000, files)
# x_test, y_test = data_provider(1)
# prediction = net.predict(path, x_test)
# 
# fig, ax = plt.subplots(1,3, figsize=(12,4))
# ax[0].imshow(x_test[0,...,0], aspect="auto")
# ax[1].imshow(y_test[0,...,1], aspect="auto")
# ax[2].imshow(prediction[0,...,1], aspect="auto")

