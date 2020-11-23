import os
import numpy as np

def trim(class_name, hz):
    for array in os.listdir(f'data/{class_name}'):
        x = np.load(f'data/{class_name}/{array}')
        x = x[:,:,:hz]
        np.save(f'data/trimmed/{array[:-4]}', x)

#trim('left', 60) 