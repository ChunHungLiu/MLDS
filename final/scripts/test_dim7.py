import cPickle as pickle
import numpy as np
import pdb
import random
import time
import csv

# load word vector
from gensim.models.word2vec import Word2Vec
wordvec_file = '../Data/glove.pruned.300d.txt'
word_vec = Word2Vec.load_word2vec_format(wordvec_file, binary=False)

import Keras_model


BATCH_SIZE = 1
MODEL_NAME = '../models/12_28_2V9.hdf5'
PREDICTION_FILE_NAME = '../predictions/12_28_2V9'
#===== load training data =====
data_prefix = '../Data/pkl/'
paths = [
    'img_q_id_test',
    'question_test',
    'choices_test',
]
#load wrods 
ID_PKL = pickle.load(open(data_prefix+paths[0]+'.pkl','rb'))
QUESTION_PKL = pickle.load(open(data_prefix+paths[1]+'.pkl','rb'))
CHOICE_PKL = pickle.load(open(data_prefix+paths[2]+'.pkl','rb'))
#load picture features
IM_ID = pickle.load(open('../Data/val2014/ID.pkl','rb'))
IM_ID_DICT = dict()
for num in xrange(len(IM_ID)):
    ID = IM_ID[num].split('_')[2].split('.')[0]
    IM_ID_DICT[ID]=num
mem_shape = (40504,1,1000)
mem_image = np.memmap('../Data/val2014/vgg_feats.memmap',dtype='float32',mode='r',shape=mem_shape )
#===== prepare pickList =====
pickList = range(0,len(ID_PKL))
numToC = {0:'A',1:'B',2:'C',3:'D',4:'E'}
answers = []
# maybe this will help?  -- Ray.
# this really help, thanks!  -- Angus.
#print '{0:{fill}{align}12}'.format(ID_PKL[0][0],fill='0',align='>')

# printing 300-dim word vector
#print word_vec[ QUESTION_PKL[0][0] ]
print "start making model..."
model = Keras_model.keras_model(20)
model.load_weights(MODEL_NAME)
#===== Start training =====
print "Start testing!"
for epoch in xrange(1):#null loop    
    tStart = time.time()
    for batch_num in xrange(0,int(len(pickList)/BATCH_SIZE)):
        start = batch_num*BATCH_SIZE    
        end   = (batch_num+1)*BATCH_SIZE
        if end > len(pickList):
            start = len(pickList)-BATCH_SIZE
            end = len(pickList)
        batched_image = []
        batched_words = []
        batched_opt1 = []
        batched_opt2 = []
        batched_opt3 = []
        batched_opt4 = []
        batched_opt5 = []
        for idx in pickList[start:end]:
            wordsList = []
            for word in QUESTION_PKL[idx]:
                if word in word_vec:
                    wordsList.append(word_vec[word])
            words = np.vstack(wordsList)
            #batched_opt1 = np.hstack([ word_vec[word] if word in word_vec else np.zeros((300,)) for word in CHOICE_PKL[idx][0] ] )
            for opt_idx,choice in enumerate(CHOICE_PKL[idx]):
                opt=[]
                for word in choice:
                    if word in word_vec:
                        opt.append(word_vec[word])
                    else:
                        opt.append(np.zeros((300,)) )
                if not opt:
                    opt.append(np.zeros((300,)) )
                # no switch in python :P
                if   opt_idx == 0:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt1.append( np.vstack(opt) )
                elif opt_idx == 1:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt2.append( np.vstack(opt) )
                elif opt_idx == 2:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt3.append( np.vstack(opt) )
                elif opt_idx == 3:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt4.append( np.vstack(opt) )
                elif opt_idx == 4:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt5.append( np.vstack(opt) )
            #Image attached
            batched_image.append(mem_image[ IM_ID_DICT[ '{0:{fill}{align}12}'.format(ID_PKL[idx][0],fill='0',align='>') ] ].flatten())
            #batched_words.append([words])
            batched_words.append(words)
        image_input = np.swapaxes( \
                        np.swapaxes( \
                            #np.asarray(batched_image),1,2
                            np.dstack(batched_image),1,2
                        ),0,1
                      )
        question_input = np.swapaxes( \
                            np.swapaxes( \
                                np.dstack(batched_words),1,2
                            ),0,1
                         )
        ans1_input = np.swapaxes( \
                        np.swapaxes( \
                            np.dstack(batched_opt1),1,2
                        ),0,1
                     )
        ans2_input = np.swapaxes( \
                        np.swapaxes( \
                            np.dstack(batched_opt2),1,2
                        ),0,1
                     )
        ans3_input = np.swapaxes( \
                        np.swapaxes( \
                            np.dstack(batched_opt3),1,2
                        ),0,1
                     )
        ans4_input = np.swapaxes( \
                        np.swapaxes( \
                            np.dstack(batched_opt4),1,2
                        ),0,1
                     )
        ans5_input = np.swapaxes( \
                        np.swapaxes( \
                            np.dstack(batched_opt5),1,2
                        ),0,1
                     )
        #pdb.set_trace()
        
        #model = Keras_model.keras_model(20)
        '''
        EJ temporarily modifies the image_input shape here, which should be (batch_size , vector=1000)
        '''
        image_new = np.array([image_input[0][0]])
        prediction = model.predict([question_input , ans1_input 
                                  , ans2_input
                                  , ans3_input
                                  , ans4_input
                                  , ans5_input , image_new ] )
        #print prediction
        answers.append( ['{0:{fill}{align}7}'.format(ID_PKL[idx][1],fill='0',align='>'),numToC[np.argmax(prediction)]] )
        print prediction
        #nn.train(image_input,question_input,ans1_input,ans2_input,ans3_input,ans4_input,ans5_input)
    tEnd = time.time()
    print "Time used:", tStart-tEnd

print "Finished testing!"

HEADER = ["q_id","ans"]
PRED_FILE = open( PREDICTION_FILE_NAME,'wb')
c = csv.writer(PRED_FILE,delimiter =',')
c.writerow(HEADER)
c.writerows(answers)

PRED_FILE.close()

