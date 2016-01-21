import cPickle
import numpy as np

def read_examples():

    MEM_PGRAM_shape = (1124823,48)
    MEM_LABEL_shape = (1124823,)
    train_len = 1124823
    MEM_DATA = 'dnn_result/posteriorgram/Angus_2.pgram'
    MEM_LABEL = 'label48.memmap'
    PKL_ID = 'ID.pkl'

    mem_pgram = np.memmap(MEM_DATA,dtype='float32',mode='r',shape=MEM_PGRAM_shape)
    mem_label = np.memmap(MEM_LABEL,dtype='int16',mode='r',shape=MEM_LABEL_shape)
    with open(PKL_ID,"rb") as filehandler:
        IDs = cPickle.load(filehandler)
    filehandler.close()

    idx = 0
    IDs_utter = []
    while idx <= len(IDs)-1:
        IDs_utter.append(["_".join(IDs[idx][0].split('_')[0:2]),IDs[idx][1]])
        #IDs_utter = [utter_name,utter_max]
        idx+=IDs[idx][1]

    data = []
    label = []
    last_pos = 0
    for ID in IDs_utter:
        sub_data = []
        sub_label = []
        for i in xrange(last_pos,last_pos+ID[1]):
            sub_data.append(mem_pgram[i].tolist())
            sub_label.append(mem_label[i].tolist())
        data.append(sub_data)
        label.append(sub_label)
        last_pos+=ID[1]

    return data,label

def read_test():
    MEM_PGRAM = 'dnn_result/posteriorgram/Angus_2_test.pgram'
    MEM_PGRAM_shape = (180406,48)
    mem_test = np.memmap(MEM_PGRAM,dtype='float32',mode='r',shape=MEM_PGRAM_shape)

    with open('ID_test.pkl',"rb") as filehandler:
        IDs = cPickle.load(filehandler)
    filehandler.close()

    idx = 0
    IDs_utter = []
    while idx <= len(IDs)-1:
        IDs_utter.append(["_".join(IDs[idx][0].split('_')[0:2]),IDs[idx][1]])
        #IDs_utter = [utter_name,utter_max]
        idx+=IDs[idx][1]

    data = []
    last_pos = 0
    for ID in IDs_utter:
        sub_data = []
        for i in xrange(last_pos,last_pos+ID[1]):
            sub_data.append(mem_test[i].tolist())
        data.append(sub_data)
        last_pos+=ID[1]
    return data,IDs_utter

