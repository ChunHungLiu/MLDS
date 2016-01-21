from collections import Counter
import cPickle as pickle
import csv
import numpy as np
import pdb
from random import shuffle

from keras.models import model_from_json

from vqa import load_data



def main():
    model_root = '../models/'
    model_json = '../models/json/'
    # Models to ensemble 
    # ( 'model_name','model_type')
    model_types = [ 
    ('elephas/overfit_noLSTM_100.h5','deep_noLSTM_1.json'),
    #('no_lstm_5layer_250.h5','noLSTM_1.json'),
    ('elephas/overfit_noLSTM_batch_100_100.h5','deep_noLSTM_1.json'),
    ('elephas/overfit_noLSTM_batch_500_80.h5','deep_noLSTM_1.json'), 
    ('elephas/overfit_noLSTM_batch_1000_60.h5','deep_noLSTM_1.json')
    ] 
     
    # Load data
    print 'Loading data...'
    X_test = load_data()[1][0]
    TEST_ID = '../Data/pkl/img_q_id_test'
    TEST_ID_PKL = pickle.load(open(TEST_ID+'.pkl','rb'))
    numoftest = X_test.shape[0] 
    # Load Models
    print 'Loading models...'
    
    models = []
    for weights, json in model_types:
        m = model_from_json(open(model_json+json).read())
        m.load_weights(model_root+weights)
        models.append(m)
    
    # Predict
    print 'Predicting...'
    grouped_answers = []
    for model in models:
        probs = model.predict(X_test,batch_size=128)
        answers= map(numToC,np.argmax(probs,axis=1).tolist())      
        grouped_answers.append(answers)

    # Ensemble
    print 'Ensembling...'
    answers = []
    for idx in xrange(numoftest):
        curlist = []
        for m in range(len(models)):
            curlist.append(grouped_answers[m][idx]) 
        max_cnt = Counter(curlist)
        m = max( v for _, v in max_cnt.iteritems())
        r = [ k for k, v in max_cnt.iteritems() if v == m ]
        shuffle(r)
        answers.append(r[0])


    # Write to CSV
    ids   = map(nameToId,[ TEST_ID_PKL[idx][1] for idx in range(len(TEST_ID_PKL)) ])
    prediction = zip(ids,answers)


    print 'Writing to CSV...'
    with open('test_ensemble.csv','wb') as fout:
        c = csv.writer(fout,delimiter =',')
        c.writerow(['q_id','ans'])
        c.writerows(prediction)
    
    print 'Done'

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

if __name__ == "__main__":
    main()
