from collections import Counter
import cPickle as pickle
import csv
import os
from random import shuffle
import pdb
import sys

ENSEMBLE_NAME = 'dnn,attention_1,2,3.csv'
TEST_ID = '../Data/pkl/img_q_id_test'
TEST_ID_PKL = pickle.load(open(TEST_ID+'.pkl','rb'))

def main():
    # has to have input files
    #if len(sys.argv) < 2:
    #    print 'Specify input files'
    
    
    print 'Reading csvs...'  
    grouped_answers = []
    question_ids = []
    #for csv_file in sys.argv[1:]:
    #files = [ x for x in os.listdir('.') if '0' in x ]
    files = ['blah_gogo.csv', 'attention2_1_8_decay_180', 'attention2_1_8_400', 'attention3_1_6_decay_60', 'no_lstm_6layer_dnn_200']
    for csv_file in files:
        print csv_file
        with open(csv_file,'r') as fin:
            spamreader = csv.reader(fin)
            next(spamreader)
            ans = [ row[1] for row in spamreader ]
            ids = [ row[0] for row in spamreader ]
            grouped_answers.append(ans)

    numoftest = len(grouped_answers[0])
    
    print 'Ensembling...'
    answers = []
    for idx in xrange(numoftest):
        curlist = []
        for m in range(len(files)):
            curlist.append(grouped_answers[m][idx])
        max_cnt = Counter(curlist)
        m = max( v for _, v in max_cnt.iteritems())
        r = [ k for k, v in max_cnt.iteritems() if v == m ]
        shuffle(r)
        answers.append(r[0])
    
    # Write to CSV
    ids = map(nameToId,[ TEST_ID_PKL[idx][1] for idx in range(len(TEST_ID_PKL)) ])
    prediction = zip(ids,answers)
     
    print 'Writing to CSV...'
    with open(ENSEMBLE_NAME,'wb') as fout:
        c = csv.writer(fout,delimiter =',')
        c.writerow(['q_id','ans'])
        c.writerows(prediction)
    
    print 'Done' 
            
def nameToId(ans_string):
    return '{0:{fill}{align}7}'.format(ans_string,fill='0',align='>')


     
if __name__ == "__main__":
    main()
