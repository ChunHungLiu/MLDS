import cPickle as pickle
import csv
import pdb

import numpy as np

import Keras_model_deep_noLSTM as Akar
from vqa import sent_load_data
from predict import nameToId, numToC

# Parameters
batch_size = 200
nb_epoch = 200
verbose = 1
validation_split = 0.1
shuffle = True
show_accuracy = True

EXP_ROOT = '../exp/'
MODEL_ROOT = '../models/elephas/'
PREDICTION_ROOT = '../predictions/'

MODEL = 'sent_40_noLSTM_batch_{}'.format(batch_size)

print 'Loading data...'
(X_train,Y_train),(X_test,Y_test) = sent_load_data()
TEST_ID = '../Data/pkl/img_q_id_test'
TEST_ID_PKL = pickle.load(open(TEST_ID+'.pkl','rb'))
ids   = map(nameToId,[ TEST_ID_PKL[idx][1] for idx in range(len(TEST_ID_PKL)) ])

print 'Building model...'
model = Akar.keras_model(1)

#print 'Defining callbacks...'
#checkpoint = ModelCheckpoint('../models/elephas/checkpoint_'+MODEL+'.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
#earlystopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0)

print 'Start training...'
#for epoch in range(1,201):
history = model.fit( X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose, callbacks=[],validation_split=validation_split, shuffle=shuffle,show_accuracy=show_accuracy)
    
acc = history.history['acc']
val_acc = history.history['val_acc']
   
logging = zip(range(1,nb_epoch+1),acc,val_acc)

log_header = ['epoch','acc','val_acc']

with open(EXP_ROOT + MODEL + '.csv','w') as fout:
    c = csv.writer(fout,delimiter=',')
    c.writerow(log_header)
    c.writerows(logging)
 
"""
    model.save_weights(MODEL_ROOT+MODEL+"_{}.h5".format(epoch))
    
    probs = model.predict(X_test,batch_size=128)

    answers = map(numToC,np.argmax(probs,axis=1).tolist())
    prediction = zip(ids,answers)
    # Write to CSV
    pred_file = PREDICTION_ROOT + MODEL + "_{}.csv".format(epoch)  

    with open(pred_file,'wb') as fout:
        c = csv.writer(fout,delimiter =',')
        c.writerow(['q_id','ans'])
        c.writerows(prediction)

    print 'Predicted {0} at epoch {1}'.format(MODEL,epoch)
"""
pdb.set_trace()
