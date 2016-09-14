require 'hdf5'
debugger = require 'fb.debugger'

loader = require 'data_loader'
printf = utils.printf

train_dir = './data/coco_inception/fea_dir_split/fea_dir_val'
fea_len = 1024
fea, fns = loader.load_dir(train_dir, fea_len)

debugger.enter()
