import os
import numpy

def dump_tensor_to_file(tensors, output_dir):
    # tensors: [name, value]
    for name, value in tensors.iteritems():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        numpy.save(os.path.join(output_dir, name), value)
