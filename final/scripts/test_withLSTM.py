import cPickle as pickle
import csv
import numpy as np
import pdb
import random
import time
#import pyprind
from progressbar import ProgressBar
# load word vector
from gensim.models.word2vec import Word2Vec
wordvec_file = '../Data/glove.pruned.300d.txt'
word_vec = Word2Vec.load_word2vec_format(wordvec_file, binary=False)

#import Keras_model_noLSTM
import Keras_model_deep_withLSTM

BATCH_SIZE = 1
MODEL_NAME = '../models/lstm_mask_ada_20.hdf5'
PREDICTION_FILE_NAME = '../predictions/lstm_mask_ada_20'
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

print "start making model..."
pdb.set_trace()
model = Keras_model_deep_withLSTM.keras_model(BATCH_SIZE)
#model = Keras_model_noLSTM.keras_model(1)
model.load_weights(MODEL_NAME)
#===== Start testing =====
print "Start testing!"
for epoch in xrange(1):#NULL loop
    tStart = time.time()
    progress = ProgressBar().start()
    for batch_num in xrange(0,int(len(pickList)/BATCH_SIZE)):
        progress.update(int((batch_num+1)/float(72801/BATCH_SIZE)*100))
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
        batched_sol  = []
        name = []
        for idx in pickList[start:end]:
            name.append(ID_PKL[idx][1])
            wordsList = []
            for word in QUESTION_PKL[idx]:
                if word in word_vec:
                    wordsList.append(word_vec[word])
            words = np.mean(wordsList, axis=0)
            n_pad = question_max_seq_len - len(wordsList)
            batched_words.append(
                np.vstack( wordsList + [np.ones((n_pad, 300))*-999] )
            )
            
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
                        n_pad = opt1_max_seq_len - len(opt)
                        batched_opt1.append( 
                            np.vstack(opt+[np.ones((n_pad, 300))*-999]) )
                        #batched_opt1.append( np.mean(opt, axis=0) )
                elif opt_idx == 1:
                    if not opt:
                        pdb.set_trace()
                    else:
                        n_pad = opt2_max_seq_len - len(opt)
                        batched_opt2.append( 
                            np.vstack(opt+[np.ones((n_pad, 300))*-999]) )
                        #batched_opt2.append( np.mean(opt, axis=0) )
                elif opt_idx == 2:
                    if not opt:
                        pdb.set_trace()
                    else:
                        n_pad = opt3_max_seq_len - len(opt)
                        batched_opt3.append( 
                            np.vstack(opt+[np.ones((n_pad, 300))*-999]) )
                        #batched_opt3.append( np.mean(opt, axis=0) )
                elif opt_idx == 3:
                    if not opt:
                        pdb.set_trace()
                    else:
                        n_pad = opt4_max_seq_len - len(opt)
                        batched_opt4.append( 
                            np.vstack(opt+[np.ones((n_pad, 300))*-999]) )
                        #batched_opt4.append( np.mean(opt, axis=0) )
                elif opt_idx == 4:
                    if not opt:
                        pdb.set_trace()
                    else:
                        n_pad = opt5_max_seq_len - len(opt)
                        batched_opt5.append( 
                            np.vstack(opt+[np.zeros((n_pad, 300))*-999]) )
                        #batched_opt5.append( np.mean(opt, axis=0) )
            #Image attached
            batched_image.append(mem_image[ IM_ID_DICT[ '{0:{fill}{align}12}'.format(ID_PKL[idx][0],fill='0',align='>') ] ].flatten())
        image_input     = np.vstack(batched_image)
        question_input  = np.dstack(batched_words)
        question_input  = np.swapaxes(question_input,0,1)
        question_input  = np.swapaxes(question_input,0,2)

        ans1_input      = np.dstack(batched_opt1)
        ans1_input      = np.swapaxes(ans1_input,0,1)
        ans1_input      = np.swapaxes(ans1_input,0,2)

        ans2_input      = np.dstack(batched_opt2)
        ans2_input      = np.swapaxes(ans2_input,0,1)
        ans2_input      = np.swapaxes(ans2_input,0,2)

        ans3_input      = np.dstack(batched_opt3)
        ans3_input      = np.swapaxes(ans3_input,0,1)
        ans3_input      = np.swapaxes(ans3_input,0,2)

        ans4_input      = np.dstack(batched_opt4)
        ans4_input      = np.swapaxes(ans4_input,0,1)
        ans4_input      = np.swapaxes(ans4_input,0,2)

        ans5_input      = np.dstack(batched_opt5)
        ans5_input      = np.swapaxes(ans5_input,0,1)
        ans5_input      = np.swapaxes(ans5_input,0,2)
        #data = np.hstack([image_input,question_input,ans1_input,ans2_input,ans3_input,ans4_input,ans5_input])
        #pdb.set_trace()
        #model = Keras_model.keras_model(20)
        prediction = model.predict_on_batch([image_input,question_input,ans1_input,ans2_input,ans3_input,ans4_input,ans5_input])
        anses =  list(np.argmax(prediction[0],axis=1))
        for (n,a) in zip(name,anses):
            answers.append( ['{0:{fill}{align}7}'.format(n,fill='0',align='>'),numToC[a]] )
        #print prediction
    tEnd = time.time()
    print "Time used:", tEnd-tStart

print "Finished testing!"


HEADER = ["q_id","ans"]
PRED_FILE = open( PREDICTION_FILE_NAME,'wb')
c = csv.writer(PRED_FILE,delimiter =',')
c.writerow(HEADER)
c.writerows(answers)

PRED_FILE.close()
