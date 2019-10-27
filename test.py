import os
from options.test_options import TestOptions
from dataloader.data_loader import dataloader
from model.models import create_model
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()
opt.model = 'test'
opt.shuffle = False

dataset = dataloader(opt)
dataset_size = len(dataset) * opt.batchSize
print ('testing images = %d ' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

# testing
for i,data in enumerate(dataset):
    model.set_input(data)
    model.test()
    model.save_results(visualizer, i*opt.batchSize)