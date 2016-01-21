import argparse
import cPickle as pickle
import csv
import numpy as np
import pdb
import sys

import Keras_model_deep_noLSTM as Akar
import vqa

(X_train,Y_train),(X_test,Y_test) = vqa.load_data()

PREDICTION_FILE_NAME = '../predictions/test_elephat_test'
MODEL_NAME = '../models/elephas/overfit_noLSTM_100.h5'

TEST_ID = '../Data/pkl/img_q_id_test'

TEST_ID_PKL = pickle.load(open(TEST_ID+'.pkl','rb'))

print "start making model..."
model = Akar.keras_model(1)
model.load_weights(MODEL_NAME)

print "Start testing..."
prediction = model._predict([X_test])

def nameToId(ans_string):
    return '{0:{fill}{align}7}'.format(ans_string,fill='0',align='>')

def numToC(ans_int):
    if ans_int == 0:
        return 'A'
    elif ans_int == 1:
        return 'B'
    elif ans_int == 2:
        return 'C'
    elif ans_int == 3:
        return 'D'
    elif ans_int == 4:
        return 'E'
    else:
        return ValueError, 'ans has to be in range(5)'

ids   = map(nameToId,[ TEST_ID_PKL[idx][1] for idx in range(len(TEST_ID_PKL)) ])
answers = map(numToC,np.argmax(prediction[0],axis=1).tolist())

pred = zip(ids,answers)

pdb.set_trace()

HEADER = ["q_id","ans"]
PRED_FILE = open( PREDICTION_FILE_NAME,'wb')
c = csv.writer(PRED_FILE,delimiter =',')
c.writerow(HEADER)
c.writerows(pred)

PRED_FILE.close()
