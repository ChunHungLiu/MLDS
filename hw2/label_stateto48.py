import numpy as np
from reader import *

MEM_LABEL = 'label.memmap'
mem_label = np.memmap(MEM_LABEL,dtype='int16',mode='r',shape=(1124823,))

dict48 = load_liststateto48()

label48 = []
for i in range(0,1124823);
