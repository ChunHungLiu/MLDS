import argparse
import cPickle as pickle
import csv
import numpy as np
import pdb

from keras.models import model_from_json

from vqa import load_data

def main():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights",type=str, required=True, help="model weights")
    parser.add_argument("-m", "--model",type=str, required=True, help="model json")
    parser.add_argument("-p", "--prediction",type=str, required=True, help="prediction file")
    args = parser.parse_args()
    
    # Load data
    print 'Loading data...'
    X_test = load_data()[1][0]
    TEST_ID = '../Data/pkl/img_q_id_test'
    TEST_ID_PKL = pickle.load(open(TEST_ID+'.pkl','rb'))
    
    # Create Model
    print 'Loading model...'
    model = model_from_json(open(args.model).read())
    model.load_weights(args.weights)
    
    # Predict
    print 'Predicting...'
    probs = model.predict(X_test,batch_size=128)
    ids   = map(nameToId,[ TEST_ID_PKL[idx][1] for idx in range(len(TEST_ID_PKL)) ])
    
    answers = map(numToC,np.argmax(probs[:,:5],axis=1).tolist())

    prediction = zip(ids,answers)

    # Write to CSV
    print 'Writing to CSV...'
    with open(args.prediction,'wb') as fout:
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
