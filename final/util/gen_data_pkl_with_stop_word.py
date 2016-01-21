import cPickle as pickle
import pdb
import re
from stop_words import get_stop_words

stop_words = get_stop_words('english')

data_prefix = './Data/final_project_pack/'
paths = {
    'annotation_train'  : 'annotation.train',
    'answer_train_sol'  : 'answer.train_sol',
    'answer_train'      : 'answer.train',
    'choices_train'     : 'choices.train',
    'choices_test'      : 'choices.test',
    'question_train'    : 'question.train',
    'question_test'     : 'question.test'
}

id_train_done = False
id_test_done = False
id_train_pkl = []
id_test_pkl = []
for k,v in paths.iteritems():
    pkl_data = []
    #ans_type = {}
    with open(data_prefix+v, 'r') as f:
        for line in f:
            remain = None
            tokens = line.split()
            if tokens[0] == 'img_id':
                continue
            img_id, q_id = int(tokens[0]), int(tokens[1])
            if 'train' in k and id_train_done is False:
                id_train_pkl.append( (img_id, q_id) )
            if 'test' in k and id_test_done is False:
                id_test_pkl.append( (img_id, q_id) )
            if k == 'question_train' or \
               k == 'question_test' or \
               k == 'answer_train':
                remain = []
                #for t in tokens[2:]:
                #if '\'not real\'' in line:
                    #pdb.set_trace()
                for t in re.findall(r"[\w']+", ' '.join(tokens[2:])):
                    t = t.lower()
                    if t in stop_words:
                        continue
                    if '\'s' in t:
                        #pdb.set_trace()
                        remain.append(t[:-2])
                        remain.append(t[-2:])
                    elif 's\'' in t:
                        remain.append(t[:-1])
                        remain.append(t[-1:])
                    elif 'n\'t' in t:
                        remain.append(t[:-3])
                        remain.append('not')
                    else:
                        for w in re.findall(r"[\w]+", t):
                            remain.append(w)
                if k != 'answer_train':
                    remain.append('?')
                #print remain
            elif k == 'answer_train_sol':
                remain = ord(tokens[2]) - ord('A')
                #print remain
            elif k == 'choices_train' or \
                 k == 'choices_test':
                tokens = re.split(r"\([A-E]\)", ' '.join(tokens[2:]) )[1:]
                remain = []
                assert len(tokens) == 5
                for tok in tokens:
                    tok = tok.lower()
                    #remain.append( re.findall(r"[\w']+", tok) )
                    sentence = []
                    for t in re.findall(r"[\w']+", tok):
                        if '\'s' in t:
                            sentence.append(t[:-2])
                            sentence.append(t[-2:])
                            #pdb.set_trace()
                        elif 's\'' in t:
                            sentence.append(t[:-1])
                            sentence.append(t[-1:])
                        elif 'n\'t' in t:
                            sentence.append(t[:-3])
                            sentence.append('not')
                        else:
                            sentence.append(t)
                    remain.append( sentence )
                #print remain
            elif k == 'annotation_train': 
                tokens = ' '.join(tokens[2:]).split('type:') 
                remain = []
                #pdb.set_trace()
                # q_type
                remain.append( tokens[1].strip('\"').split('\"')[0] )
                # a_type
                ans = tokens[2].strip('\"')
                ''' 
                if ans not in ans_type:
                    ans_type[ans] = 0
                else:
                    ans_type[ans] += 1
                '''
                if ans == 'number':
                    remain.append(1)
                else:
                    remain.append(0)
                #print remain
                #print q_id
            pkl_data.append(remain)
        '''if k=='annotation_train':
            pdb.set_trace()
        '''
    if len(id_train_pkl) != 0 and id_train_done is False:
        if len(id_train_pkl) != len(pkl_data):
            pdb.set_trace()
        fh = open('./Data/pkl/img_q_id_train.pkl','wb')
        pickle.dump(id_train_pkl, fh, -1)
        fh.close()
        id_train_done = True
    if len(id_test_pkl) != 0 and id_test_done is False:
        if len(id_test_pkl) != len(pkl_data):
            pdb.set_trace()
        fh = open('./Data/pkl/img_q_id_test.pkl','wb')
        pickle.dump(id_test_pkl, fh, -1)
        fh.close()
        id_test_done = True
    fh = open('./Data/pkl/'+k+'_with_stop_word.pkl','wb')
    pickle.dump(pkl_data, fh, -1)
    fh.close()
