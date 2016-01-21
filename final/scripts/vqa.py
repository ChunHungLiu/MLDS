import cPickle as pickle
import os.path
import pdb
import random

import h5py
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
import numpy as np

"""
Usage:
    (X_train, Y_train), (X_test, Y_test) = vqa.load_data()
"""
###########################
#        Data File        #
###########################

DATA_FILE = '../Data/vqa_dim_2800.h5'
SENT_DATA_FILE = '../Data/sent_vqa_dim_2800.h5'
ANNOTATION_DATA_FILE = '../Data/annotation_vqa_dim_2800.h5'
QTYPE_DATA_FILE = '../Data/qtype_vqa_dim_2800.h5'
STOP_DATA_FILE = '../Data/stop_vqa_dim_2800.h5'

###########################
#        Data paths       #
###########################
data_prefix = '../Data/pkl/'
stop_prefix = '_with_stop_word'
train_paths = [
    'img_q_id_train',
    'question_train',
    'choices_train',
    'answer_train',
    'answer_train_sol',
    'annotation_train'
]

test_paths = [
    'img_q_id_test',
    'question_test',
    'choices_test'
]

ID_PKL = pickle.load(open(data_prefix+train_paths[0]+'.pkl','rb'))
QUESTION_PKL = pickle.load(open(data_prefix+train_paths[1]+'.pkl','rb'))
CHOICE_PKL = pickle.load(open(data_prefix+train_paths[2]+'.pkl','rb'))
#ANSWER_PKL = pickle.load(open(data_prefix+train_paths[3]+'.pkl','rb'))
ANS_SOL_PKL = pickle.load(open(data_prefix+train_paths[4]+'.pkl','rb'))
#ANNOTATION_PKL = pickle.load(open(data_prefix+train_paths[5]+'.pkl','rb'))

TEST_ID_PKL = pickle.load(open(data_prefix+test_paths[0]+'.pkl','rb'))
TEST_QUESTION_PKL = pickle.load(open(data_prefix+test_paths[1]+'.pkl','rb'))
TEST_CHOICE_PKL = pickle.load(open(data_prefix+test_paths[2]+'.pkl','rb'))

#stop word pkls
STOP_QUESTION_PKL = pickle.load(open(data_prefix+train_paths[1]+stop_prefix+'.pkl','rb'))
STOP_CHOICE_PKL = pickle.load(open(data_prefix+train_paths[2]+stop_prefix+'.pkl','rb'))
#STOP_ANSWER_PKL = pickle.load(open(data_prefix+train_paths[3]+stop_prefix+'.pkl','rb'))
STOP_ANS_SOL_PKL = pickle.load(open(data_prefix+train_paths[4]+stop_prefix+'.pkl','rb'))
#STOP_ANNOTATION_PKL = pickle.load(open(data_prefix+train_paths[5]+stop_prefix+'.pkl','rb'))

#load wrods
STOP_TEST_QUESTION_PKL = pickle.load(open(data_prefix+test_paths[1]+stop_prefix+'.pkl','rb'))
STOP_TEST_CHOICE_PKL = pickle.load(open(data_prefix+test_paths[2]+stop_prefix+'.pkl','rb'))

###########################
#      Image Features     #
###########################
IM_ID = pickle.load(open('../Data/train2014/ID.pkl','rb'))
IM_ID_DICT = dict()
for num in xrange(len(IM_ID)):
    ID = IM_ID[num].split('_')[2].split('.')[0]
    IM_ID_DICT[ID]=num
mem_shape = (82783,1,1000)
mem_image = np.memmap('../Data/train2014/vgg_feats.memmap',dtype='float32',mode='r',shape=mem_shape )


TEST_IM_ID = pickle.load(open('../Data/val2014/ID.pkl','rb'))
TEST_IM_ID_DICT = dict()
for num in xrange(len(TEST_IM_ID)):
    TEST_ID = TEST_IM_ID[num].split('_')[2].split('.')[0]
    TEST_IM_ID_DICT[TEST_ID] = num
test_mem_shape = (40504,1,1000)
test_mem_image = np.memmap('../Data/val2014/vgg_feats.memmap',dtype='float32',mode='r',shape=test_mem_shape)

def save_data():
    """
        Packs data to pickle
    """
    ###########################
    #        Word 2 Vec       #
    ###########################
    wordvec_file = '../Data/glove.pruned.300d.txt'
    word_vec = Word2Vec.load_word2vec_format(wordvec_file, binary=False)

    def word2vec(word):
        if word in word_vec:
            return word_vec[word]
        else:
            return np.zeros((300,))

    X_train = []
    Y_train = []
    for idx in range(0,len(ID_PKL)):
        # Image
        img_vec = mem_image[ IM_ID_DICT[ '{0:{fill}{align}12}'.format(ID_PKL[idx][0],fill='0',align='>') ] ].flatten()

        # Question
        question_vec = np.mean(map(word2vec,QUESTION_PKL[idx]),axis=0)

        # Options
        cur_choices = [ choice if choice else ['!'] for choice in CHOICE_PKL[idx] ]
        opts_vec = np.concatenate( tuple(\
                    [ np.mean( map(word2vec,choice),axis = 0 ) for choice in cur_choices ]\
                    ),axis=0)

        # Solution
        Sol_vec = np.asarray([0. if x != ANS_SOL_PKL[idx] else 1. for x in range(5) ])

        # A (2800,) vector
        Feat_vec = np.concatenate((img_vec,question_vec,opts_vec),axis=0)

        X_train.append(Feat_vec)
        Y_train.append(Sol_vec)

    print 'Processed training data'

    #pdb.set_trace()

    X_test = []
    Y_test = []
    for idx in range(0,len(TEST_ID_PKL)):
        img_vec = test_mem_image[ TEST_IM_ID_DICT[ '{0:{fill}{align}12}'.format(TEST_ID_PKL[idx][0],fill='0',align='>') ] ].flatten()

        # Question
        question_vec = np.mean(map(word2vec,TEST_QUESTION_PKL[idx]),axis=0)

        # Options
        cur_choices = [ choice if choice else ['!'] for choice in TEST_CHOICE_PKL[idx] ]
        opts_vec = np.concatenate( tuple(\
                    [ np.mean( map(word2vec,choice),axis = 0 ) for choice in cur_choices ]\
                    ),axis=0)
        # Solution
        # Test Set has no Solutions

        # A (2800,) vector
        Feat_vec = np.concatenate((img_vec,question_vec,opts_vec),axis=0)

        X_test.append(Feat_vec)


    print 'Processed testing data'

    X_train = np.vstack(X_train)
    Y_train = np.vstack(Y_train)

    X_test = np.vstack(X_test)

    h5f = h5py.File(DATA_FILE, 'w')
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('Y_train', data=Y_train)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('Y_test', data=Y_test)
    h5f.close()


def sent_save_data():
    """
        Packs data to pickle
    """
    ###########################
    #        Doc 2 Vec        #
    ###########################

    doc2vec_file = '../Data/sent_vec_1B_benchmark_20_files.doc2vec'
    doc_vec = Doc2Vec.load(doc2vec_file)

    print 'Doc2vec model loaded'

    X_train = []
    Y_train = []
    for idx in range(0,len(ID_PKL)):
        # Image
        img_vec = mem_image[ IM_ID_DICT[ '{0:{fill}{align}12}'.format(ID_PKL[idx][0],fill='0',align='>') ] ].flatten()

        # Question
        question_vec = doc_vec.docvecs['question_train_{}'.format(idx)]

        # Options
        opts_vec = np.concatenate( tuple(\
                    [ doc_vec.docvecs['choices_train_{0}_{1}'.format(idx,choice_idx)] \
                        for choice_idx in range(5) ]\
                    ),axis=0)

        # Solution
        Sol_vec = np.asarray([0. if x != ANS_SOL_PKL[idx] else 1. for x in range(5) ])

        # A (2800,) vector
        Feat_vec = np.concatenate((img_vec,question_vec,opts_vec),axis=0)

        X_train.append(Feat_vec)
        Y_train.append(Sol_vec)

    print 'Processed training data'

    pdb.set_trace()

    X_test = []
    Y_test = []
    for idx in range(0,len(TEST_ID_PKL)):
        img_vec = test_mem_image[ TEST_IM_ID_DICT[ '{0:{fill}{align}12}'.format(TEST_ID_PKL[idx][0],fill='0',align='>') ] ].flatten()

        # Questions
        question_vec = doc_vec.docvecs['question_test_{}'.format(idx)]

        # Options
        opts_vec = np.concatenate( tuple(\
                    [ doc_vec.docvecs['choices_test_{0}_{1}'.format(idx,choice_idx)] \
                        for choice_idx in range(5) ]\
                    ),axis=0)

        # Solution
        # Test Set has no Solutions

        # A (2800,) vector
        Feat_vec = np.concatenate((img_vec,question_vec,opts_vec),axis=0)

        X_test.append(Feat_vec)


    print 'Processed testing data'

    X_train = np.vstack(X_train)
    Y_train = np.vstack(Y_train)

    X_test = np.vstack(X_test)

    h5f = h5py.File(SENT_DATA_FILE, 'w')
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('Y_train', data=Y_train)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('Y_test', data=Y_test)
    h5f.close()


def annotation_save_data():
    """
        Packs data to pickle
    """
    ###########################
    #        Word 2 Vec       #
    ###########################
    wordvec_file = '../Data/glove.pruned.300d.txt'
    word_vec = Word2Vec.load_word2vec_format(wordvec_file, binary=False)

    def word2vec(word):
        if word in word_vec:
            return word_vec[word]
        else:
            return np.zeros((300,))

    X_train = []
    Y_train = []
    for idx in range(0,len(ID_PKL)):
        # Image
        img_vec = mem_image[ IM_ID_DICT[ '{0:{fill}{align}12}'.format(ID_PKL[idx][0],fill='0',align='>') ] ].flatten()

        # Question
        question_vec = np.mean(map(word2vec,QUESTION_PKL[idx]),axis=0)

        # Options
        cur_choices = [ choice if choice else ['!'] for choice in CHOICE_PKL[idx] ]
        opts_vec = np.concatenate( tuple(\
                    [ np.mean( map(word2vec,choice),axis = 0 ) for choice in cur_choices ]\
                    ),axis=0)

        # Solution
        ans_vec = np.asarray([0. if x != ANS_SOL_PKL[idx] else 1. for x in range(5) ])

        annotation_vec = np.mean( map(word2vec,ANNOTATION_PKL[idx][0].split()),axis=0)

        question_type_vec = np.asarray( [ ANNOTATION_PKL[idx][1] ] )

        Sol_vec = np.concatenate((ans_vec,annotation_vec,question_type_vec),axis=0)

        # A (2800,) vector
        Feat_vec = np.concatenate((img_vec,question_vec,opts_vec),axis=0)

        X_train.append(Feat_vec)
        Y_train.append(Sol_vec)

    print 'Processed training data'

    #pdb.set_trace()

    X_test = []
    Y_test = []
    for idx in range(0,len(TEST_ID_PKL)):
        img_vec = test_mem_image[ TEST_IM_ID_DICT[ '{0:{fill}{align}12}'.format(TEST_ID_PKL[idx][0],fill='0',align='>') ] ].flatten()

        # Question
        question_vec = np.mean(map(word2vec,TEST_QUESTION_PKL[idx]),axis=0)

        # Options
        cur_choices = [ choice if choice else ['!'] for choice in TEST_CHOICE_PKL[idx] ]
        opts_vec = np.concatenate( tuple(\
                    [ np.mean( map(word2vec,choice),axis = 0 ) for choice in cur_choices ]\
                    ),axis=0)
        # Solution
        # Test Set has no Solutions

        # A (2800,) vector
        Feat_vec = np.concatenate((img_vec,question_vec,opts_vec),axis=0)

        X_test.append(Feat_vec)


    print 'Processed testing data'

    X_train = np.vstack(X_train)
    Y_train = np.vstack(Y_train)

    X_test = np.vstack(X_test)

    h5f = h5py.File(ANNOTATION_DATA_FILE, 'w')
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('Y_train', data=Y_train)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('Y_test', data=Y_test)
    h5f.close()


def qtype_save_data():
    """
        Packs data to pickle
    """
    ###########################
    #        Word 2 Vec       #
    ###########################
    wordvec_file = '../Data/glove.pruned.300d.txt'
    word_vec = Word2Vec.load_word2vec_format(wordvec_file, binary=False)

    def word2vec(word):
        if word in word_vec:
            return word_vec[word]
        else:
            return np.zeros((300,))

    X_train = []
    Y_train = []
    for idx in range(0,len(ID_PKL)):
        # Image
        img_vec = mem_image[ IM_ID_DICT[ '{0:{fill}{align}12}'.format(ID_PKL[idx][0],fill='0',align='>') ] ].flatten()

        # Question
        question_vec = np.mean(map(word2vec,QUESTION_PKL[idx]),axis=0)

        # Options
        cur_choices = [ choice if choice else ['!'] for choice in CHOICE_PKL[idx] ]
        opts_vec = np.concatenate( tuple(\
                    [ np.mean( map(word2vec,choice),axis = 0 ) for choice in cur_choices ]\
                    ),axis=0)

        # Solution
        ans_vec = np.asarray([0. if x != ANS_SOL_PKL[idx] else 1. for x in range(5) ])

        #annotation_vec = np.mean( map(word2vec,ANNOTATION_PKL[idx][0].split()),axis=0)

        question_type_vec = np.asarray( [ ANNOTATION_PKL[idx][1] ] )

        Sol_vec = np.concatenate((ans_vec,question_type_vec),axis=0)

        # A (2800,) vector
        Feat_vec = np.concatenate((img_vec,question_vec,opts_vec),axis=0)

        X_train.append(Feat_vec)
        Y_train.append(Sol_vec)

    print 'Processed training data'

    pdb.set_trace()

    X_test = []
    Y_test = []
    for idx in range(0,len(TEST_ID_PKL)):
        img_vec = test_mem_image[ TEST_IM_ID_DICT[ '{0:{fill}{align}12}'.format(TEST_ID_PKL[idx][0],fill='0',align='>') ] ].flatten()

        # Question
        question_vec = np.mean(map(word2vec,TEST_QUESTION_PKL[idx]),axis=0)

        # Options
        cur_choices = [ choice if choice else ['!'] for choice in TEST_CHOICE_PKL[idx] ]
        opts_vec = np.concatenate( tuple(\
                    [ np.mean( map(word2vec,choice),axis = 0 ) for choice in cur_choices ]\
                    ),axis=0)
        # Solution
        # Test Set has no Solutions

        # A (2800,) vector
        Feat_vec = np.concatenate((img_vec,question_vec,opts_vec),axis=0)

        X_test.append(Feat_vec)


    print 'Processed testing data'

    X_train = np.vstack(X_train)
    Y_train = np.vstack(Y_train)

    X_test = np.vstack(X_test)

    h5f = h5py.File(QTYPE_DATA_FILE, 'w')
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('Y_train', data=Y_train)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('Y_test', data=Y_test)
    h5f.close()


def stop_save_data():
    """
        Packs data to pickle
    """
    ###########################
    #        Word 2 Vec       #
    ###########################
    wordvec_file = '../Data/glove.pruned.300d.txt'
    word_vec = Word2Vec.load_word2vec_format(wordvec_file, binary=False)

    def word2vec(word):
        if word in word_vec:
            return word_vec[word]
        else:
            return np.zeros((300,))

    X_train = []
    Y_train = []
    for idx in range(0,len(ID_PKL)):
        # Image
        img_vec = mem_image[ IM_ID_DICT[ '{0:{fill}{align}12}'.format(ID_PKL[idx][0],fill='0',align='>') ] ].flatten()

        # Question
        question_vec = np.mean(map(word2vec,STOP_QUESTION_PKL[idx]),axis=0)

        # Options
        cur_choices = [ choice if choice else ['!'] for choice in STOP_CHOICE_PKL[idx] ]
        opts_vec = np.concatenate( tuple(\
                    [ np.mean( map(word2vec,choice),axis = 0 ) for choice in cur_choices ]\
                    ),axis=0)

        # Solution
        Sol_vec = np.asarray([0. if x != ANS_SOL_PKL[idx] else 1. for x in range(5) ])

        # A (2800,) vector
        Feat_vec = np.concatenate((img_vec,question_vec,opts_vec),axis=0)

        X_train.append(Feat_vec)
        Y_train.append(Sol_vec)

    print 'Processed training data'

    #pdb.set_trace()

    X_test = []
    Y_test = []
    for idx in range(0,len(TEST_ID_PKL)):
        img_vec = test_mem_image[ TEST_IM_ID_DICT[ '{0:{fill}{align}12}'.format(TEST_ID_PKL[idx][0],fill='0',align='>') ] ].flatten()

        # Question
        question_vec = np.mean(map(word2vec,STOP_TEST_QUESTION_PKL[idx]),axis=0)

        # Options
        cur_choices = [ choice if choice else ['!'] for choice in STOP_TEST_CHOICE_PKL[idx] ]
        opts_vec = np.concatenate( tuple(\
                    [ np.mean( map(word2vec,choice),axis = 0 ) for choice in cur_choices ]\
                    ),axis=0)
        # Solution
        # Test Set has no Solutions

        # A (2800,) vector
        Feat_vec = np.concatenate((img_vec,question_vec,opts_vec),axis=0)

        X_test.append(Feat_vec)


    print 'Processed testing data'

    X_train = np.vstack(X_train)
    Y_train = np.vstack(Y_train)

    X_test = np.vstack(X_test)

    h5f = h5py.File(STOP_DATA_FILE, 'w')
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('Y_train', data=Y_train)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('Y_test', data=Y_test)
    h5f.close()



def load_data():
    """
    Usage:
        (X_train, Y_train), (X_test, Y_test) = vqa.load_data()
    """
    if not os.path.isfile(DATA_FILE):
        save_data()

    h5f = h5py.File(DATA_FILE,'r')
    X_train = h5f['X_train'][:]
    Y_train = h5f['Y_train'][:]
    X_test = h5f['X_test'][:]
    Y_test = h5f['Y_test'][:]
    h5f.close()

    data = ((X_train, Y_train), (X_test, Y_test))

    return data

def sent_load_data():
    """
    Usage:
        (X_train, Y_train), (X_test, Y_test) = vqa.load_data()
    """
    if not os.path.isfile(DATA_FILE):
        save_data()

    h5f = h5py.File(SENT_DATA_FILE,'r')
    X_train = h5f['X_train'][:]
    Y_train = h5f['Y_train'][:]
    X_test = h5f['X_test'][:]
    Y_test = h5f['Y_test'][:]
    h5f.close()

    data = ((X_train, Y_train), (X_test, Y_test))

    return data


def annotation_load_data():
    """
    Usage:
        (X_train, Y_train), (X_test, Y_test) = vqa.load_data()
    """
    if not os.path.isfile(ANNOTATION_DATA_FILE):
        save_data()

    h5f = h5py.File(ANNOTATION_DATA_FILE,'r')
    X_train = h5f['X_train'][:]
    Y_train = h5f['Y_train'][:]
    X_test = h5f['X_test'][:]
    Y_test = h5f['Y_test'][:]
    h5f.close()

    data = ((X_train, Y_train), (X_test, Y_test))

    return data

def qtype_load_data():
    """
    Usage:
        (X_train, Y_train), (X_test, Y_test) = vqa.load_data()
    """
    if not os.path.isfile(QTYPE_DATA_FILE):
        save_data()

    h5f = h5py.File(QTYPE_DATA_FILE,'r')
    X_train = h5f['X_train'][:]
    Y_train = h5f['Y_train'][:]
    X_test = h5f['X_test'][:]
    Y_test = h5f['Y_test'][:]
    h5f.close()

    data = ((X_train, Y_train), (X_test, Y_test))

    return data


def stop_load_data():
    """
    Usage:
        (X_train, Y_train), (X_test, Y_test) = vqa.load_data()
    """
    if not os.path.isfile(STOP_DATA_FILE):
        save_data()

    h5f = h5py.File(STOP_DATA_FILE,'r')
    X_train = h5f['X_train'][:]
    Y_train = h5f['Y_train'][:]
    X_test = h5f['X_test'][:]
    Y_test = h5f['Y_test'][:]
    h5f.close()

    data = ((X_train, Y_train), (X_test, Y_test))

    return data


if __name__ == "__main__":
    stop_save_data()
    pdb.set_trace()
    (X_train, Y_train), (X_test, Y_test) = stop_load_data()
