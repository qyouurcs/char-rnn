#!/usr/bin/python
'''
All because torch hdf5 does not support the string format, which is really annoying.
'''

import os
import sys
import h5py
import glob

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print 'Usage: {0} <fea_dir> <save_dir>'.format(sys.argv[0])
        sys.exit()

    fea_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for h5_fn in glob.glob(os.path.join(fea_dir, '*')):
        hid = h5py.File(h5_fn)
        fns = hid['fns'][:]
        feas = hid['fea'][:]
        hid.close()
    
        base_fn = os.path.basename(h5_fn)
        base_fn = os.path.splitext(base_fn)[0]
        save_fn = os.path.join(save_dir, base_fn)
        with open(save_fn,'w') as fid:
            for fn in fns:
                print >>fid, fn
        
        save_fn = os.path.join(save_dir,os.path.basename(h5_fn))
        # now it's the fea, we just save it to a h5 file.

        f = h5py.File(save_fn,'w')
        f.create_dataset('fea', data = feas)
        f.close()
    
    print 'Done with', save_dir
