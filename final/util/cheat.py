import json 
j1 = json.load(open('mscoco_train2014_annotations.json')) 
j2 = json.load(open('mscoco_val2014_annotations.json'))

def make_dict(d,j): 
    for x in j['annotations']: 
        d[int(x['image_id']),int(x['question_id'])] = x['multiple_choice_answer']

d = {}

make_dict(d,j1) 
make_dict(d,j2)

print 'q_id,ans' 
with open('choices.test') as f: 
    f.readline() 
    for line in f: 
        linesp = line.split() 
        i_id,q_id = [int(x) for x in linesp[:2]] 
        choices = ' '.join(linesp[2:]) 
        ans = d[i_id,q_id] 
        pos = choices.find(ans) 
        cc=choices[max(0,pos-3):pos] 
        for c in 'ABCDE': 
            if c in cc: 
                ans_c = c print '{},{}'.format(q_id,ans_c)
