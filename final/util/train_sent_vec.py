from collections import namedtuple
import cPickle as pickle
import os
import pdb
from random import shuffle
import sys
import time

import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models.doc2vec import LabeledSentence as LS
from gensim.utils import to_unicode
# Having only 32G RAM, we cannot train with all of the 1 billion words


MODEL_NAME = 'sent_vec_1B_benchmark_20_files'
# Background Path
# 1-billion-benchmark
background_prefix = '../Data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled'

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

sentences = []
# Read Background sentences
count = 0
file_count = 0
for f_name in os.listdir(background_prefix):
    print f_name
    with open(background_prefix+'/'+f_name,'r') as f:
        for line in f:
            words = to_unicode(line.strip('\n')).split()
            tags = ['SENT_{}'.format(count)]
            sentences.append(LS(words=words,tags=tags))
            count += 1
    file_count += 1
    print 'Read {} background files'.format(file_count)
    if file_count == 20:
        break
# Read Documents
for path in paths:
    raw_data = pickle.load( open( data_prefix + path + pkl_prefix, 'rb' ) )
    for idx in range(len(raw_data)):
        if path == "annotation_train":
            words = to_unicode(raw_data[idx][0]).split()
            tags = [ "annotation_train_{}".format(idx) ]
            sentences.append(LS(words=words,tags=tags))

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

    print "Done getting {}'s sentences".format(path)

# Initialize Model
print "Initializing model..."
model = Doc2Vec(alpha=0.025,min_alpha=0.025,min_count=5,workers=4)

print 'Building vocabulary...'
model.build_vocab(sentences)

print "Start training..."
for epoch in range(40):
    tStart = time.time()
    shuffle(sentences)
    model.train(sentences)
    if epoch < 20:
        model.alpha -= 0.001
        model.min_alpha = model.alpha
    tEnd = time.time()

    t = tEnd - tStart
    print "Trained {0} epoch out of 40 epochs in {1} seconds".format(epoch+1,t)


print "Done training, haha!"
pdb.set_trace()

model.save(model_prefix + MODEL_NAME + doc2vec_prefix)
