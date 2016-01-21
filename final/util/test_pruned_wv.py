import cPickle as pickle
import time
import pdb

from gensim.models.word2vec import Word2Vec

wordvec_file = './Data/glove.pruned.300d.txt'

tStart = time.time()
word_vec = Word2Vec.load_word2vec_format(wordvec_file, binary=False)
tEnd = time.time()
print "Loading word vector : ",tEnd-tStart

#===== load training data =====
data_prefix = './Data/pkl/'
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

QUESTION_PKL = pickle.load(open(data_prefix+paths[1]+'.pkl','rb'))
CHOICE_PKL = pickle.load(open(data_prefix+paths[2]+'.pkl','rb'))

for choices in CHOICE_PKL:
    for sentence in choices:
        for word in sentence:
            if 'taco' in word:
                pdb.set_trace()
pdb.set_trace()
