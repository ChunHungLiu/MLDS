import cPickle as pickle
import pdb

from gensim.models.word2vec import Word2Vec
wordvec_file = './Data/glove.pruned.300d.txt'
word_vec = Word2Vec.load_word2vec_format(wordvec_file, binary=False)

data_prefix = './Data/pkl/'
paths = [
    'img_q_id_train',
    'question_train',
    'choices_train',
    'answer_train',
    'answer_train_sol',
    'annotation_train',
]

QUESTION_PKL = pickle.load(open(data_prefix+paths[1]+'.pkl','rb'))
CHOICE_PKL = pickle.load(open(data_prefix+paths[2]+'.pkl','rb'))

