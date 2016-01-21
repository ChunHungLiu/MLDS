from collections import namedtuple
import cPickle as pickle
import multiprocessing
import pdb
import sys
from random import shuffle

import gensim
from gensim.models import Doc2Vec
#from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import LabeledSentence as LS
from gensim.utils import to_unicode

# Pre_trained Word Vectors
pre_trained = '../Data/glove.840B.300d.txt'
#pre_trained = '../Data/glove.pruned.300d.txt'

# Data Paths
model_prefix = '../Data/'
data_prefix = '../Data/pkl/'
paths = [
    'annotation_train',
    #'answer_train_sol',
    'answer_train',
    'choices_train',
    'choices_test',
    'question_train',
    'question_test'
]

pkl_prefix = '.pkl'

doc2vec_prefix = '.doc2vec'

dm = 1


sentences = []
total_sentence_count = []
# Read Documents
for path in paths:
    raw_data = pickle.load( open( data_prefix + path + pkl_prefix, 'rb' ) )
    for idx in range(len(raw_data)):
        if path == "annotation_train":
            words = to_unicode(raw_data[idx][0]).split()
            tags = [ to_unicode( path + "_%s" % idx ) ]
            sentences.append(LS(words,tags))

        elif path in ["answer_train","question_train","question_test"]:
            words = map(to_unicode,raw_data[idx])
            tags = [ to_unicode( path + "_%s" % idx ) ]
            sentences.append(LS(words,tags))

        elif path in ["choices_train", "choices_test"]:
            for sub_idx in range(len(raw_data[idx])):
                words = map(to_unicode,raw_data[idx][sub_idx])
                tags = [ to_unicode( path + "_%s_%s" % (idx,sub_idx)  ) ]
                sentences.append(LS(words,tags))
        else:
            sys.exit("Something wrong!")

    if path not in ["choices_train", "choices_test"]:
        total_sentence_count.append(len(raw_data))
    else:
        total_sentence_count.append(5 * len(raw_data))

    print "Done getting {}'s sentences".format(path)

assert len(sentences) == sum(total_sentence_count)



# Initialize Model
print "Initializing model..."
cores = multiprocessing.cpu_count()

model = Doc2Vec(dm=dm, dm_concat=1, size=300, window=5, negative=5, hs=0, min_count=2, workers=cores)

model.build_vocab(sentences)

#pdb.set_trace()

model.intersect_word2vec_format(pre_trained, binary=False)  # C binary format

model.alpha = 0.025

print "Start training..."

for epoch in range(20):
    shuffle(sentences)
    model.train(sentences)
    model.alpha -= 0.001
    model.min_alpha = model.alpha
    print "Trained {0} epoch out of 20 epochs".format(epoch)

MODEL = model_prefix + pre_trained + "_" + ( "PV-DM" if dm else "PV-DBOW") + "_" + doc2vec_prefix

model.save(MODEL)

print "Done training, haha!"
