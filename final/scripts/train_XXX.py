import cPickle as pickle
import numpy as np
import pdb
import random
import time
# load word vector
from gensim.models.word2vec import Word2Vec
wordvec_file = '../Data/glove.pruned.300d.txt'
word_vec = Word2Vec.load_word2vec_format(wordvec_file, binary=False)

BATCH_SIZE = 1
MAX_EPOCH = 20
VAL_SET_RATIO = 0.9

#===== load training data =====
data_prefix = '../Data/pkl/'
paths = [
    'img_q_id_train',
    'question_train',
    'choices_train',
    'answer_train',
    'answer_train_sol',
    'annotation_train',
    #'img_q_id_test',
    #'question_test',
    #'choices_test'
]
#load wrods 
ID_PKL = pickle.load(open(data_prefix+paths[0]+'.pkl','rb'))
QUESTION_PKL = pickle.load(open(data_prefix+paths[1]+'.pkl','rb'))
CHOICE_PKL = pickle.load(open(data_prefix+paths[2]+'.pkl','rb'))
ANSWER_PKL = pickle.load(open(data_prefix+paths[3]+'.pkl','rb'))
ANS_SOL_PKL = pickle.load(open(data_prefix+paths[4]+'.pkl','rb'))
ANNO_PKL = pickle.load(open(data_prefix+paths[5]+'.pkl','rb'))
#load picture features
IM_ID = pickle.load(open('../Data/train2014/ID.pkl','rb'))
IM_ID_DICT = dict()
for num in xrange(len(IM_ID)):
    ID = IM_ID[num].split('_')[2].split('.')[0]
    IM_ID_DICT[ID]=num
mem_shape = (82783,1,1000)
mem_image = np.memmap('../Data/train2014/vgg_feats.memmap',dtype='float32',mode='r',shape=mem_shape )
#===== prepare pickList =====
pickList = range(0,len(ID_PKL))
random.shuffle(pickList)

# maybe this will help?  -- Ray.
# this really help, thanks!  -- Angus.
# print '{0:{fill}{align}12}'.format(ID_PKL[0][0],fill='0',align='>')

# printing 300-dim word vector
# print word_vec[ QUESTION_PKL[0][0] ]

#===== Start training =====
print "Start training!"
for epoch in xrange(MAX_EPOCH):
    tStart = time.time()
    for batch_num in xrange(0,int(len(pickList)/BATCH_SIZE)):
        print batch_num
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
        batched_sol = []
        for idx in pickList[start:end]:
            print ID_PKL[idx]
            words = np.vstack([ 
                    word_vec[word] \
                    if word in word_vec \
                    else np.zeros((300,)) \
                    for word in QUESTION_PKL[idx] 
            ])
            batched_opt1 = np.hstack([ word_vec[word] if word in word_vec else np.zeros((300,)) for word in CHOICE_PKL[idx][0] ] )
            batched_opt2 = np.hstack([ word_vec[word] if word in word_vec else np.zeros((300,)) for word in CHOICE_PKL[idx][1] ] )
            batched_opt3 = np.hstack([ word_vec[word] if word in word_vec else np.zeros((300,)) for word in CHOICE_PKL[idx][2] ] )
            batched_opt4 = np.hstack([ word_vec[word] if word in word_vec else np.zeros((300,)) for word in CHOICE_PKL[idx][3] ] )
            batched_opt5 = np.hstack([ word_vec[word] if word in word_vec else np.zeros((300,)) for word in CHOICE_PKL[idx][4] ] )
            #batched_sol  = np.
            #Image attached
            batched_image.append(mem_image[ IM_ID_DICT[ '{0:{fill}{align}12}'.format(ID_PKL[idx][0],fill='0',align='>') ] ].flatten())
            batched_words.append([words])
        image_input = np.asarray(batched_image)
        question_input = np.dstack(batched_words)
        ans1_input = np.dstack(batched_opt1)
        ans2_input = np.dstack(batched_opt2)
        ans3_input = np.dstack(batched_opt3)
        ans4_input = np.dstack(batched_opt4)
        ans5_input = np.dstack(batched_opt5)
        pdb.set_trace()
