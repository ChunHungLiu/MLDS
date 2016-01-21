import cPickle as pickle
import pdb

from gensim.models.word2vec import Word2Vec

prefix = "./Data/pkl/"
paths = [
    'choices_train',
    'choices_test',
    'question_train',
    'question_test'
]
wordvec_file = './Data/glove.840B.300d.txt'
pruned_wordvec_file = './Data/glove.pruned.300d.txt'

word_vec = Word2Vec.load_word2vec_format(wordvec_file, binary=False)
#pdb.set_trace()

word_count = {}
for p in paths:
    data = pickle.load(open(prefix+p+'.pkl','rb'))
    if 'question' in p:
        for d in data:
            for sentence in d:
                #pdb.set_trace()
                for word in sentence.split():
                    word = word.strip(',')
                    if word in word_vec:
                        if word in word_count:
                            word_count[word] += 1
                        else:
                            word_count[word] = 1
    else:
        for choices in data:
            for sentence in choices:
                for word in sentence:
                    if word in word_vec:
                        if word in word_count:
                            word_count[word] += 1
                        else:
                            word_count[word] = 1
                    
#pdb.set_trace()
with open(wordvec_file, 'r') as in_f, \
     open(pruned_wordvec_file, 'w') as out_f:
    for line in in_f:
        tokens = line.split()
        if len(tokens) == 2:
            continue
        if tokens[0] in word_count:
            out_f.write(line)
