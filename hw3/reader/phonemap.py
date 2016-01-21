import numpy as np

mapdir = './Data/phones/48_39.map'
mapdir2 = './Data/phones/state_48_39.map'
map48_chr='./Data/phones/48_idx_chr.map'
'''
def get_PhoneDict(num = 48):
    if num != 48 and num != 39:
        print "Phone Number is 48 or 39!"
        return

    phonemap = []
    with open(mapdir) as fin:
        for line in fin:
            pair = line.split()
            if num == 48:
                phonemap.append(pair[0])
            elif num == 39:
                phonemap.append(pair[1])

    p = dict()
    for i in range(len(phonemap)):
        Feat = [0]*len(phonemap)
        Feat[i] = 1
        p[phonemap[i]] = Feat

    return p
'''
def get_PhoneStateDict():
    phonemap = []
    with open(mapdir2) as fin:
        for line in fin:
            pair = line.split()
            phonemap.append(pair[0])

    p = dict()
    for i in range(len(phonemap)):
        Feat = [0]*len(phonemap)
        Feat[i] = 1
        p[i] = Feat

    return p

def get_PhoneStateVec():
    phonemap = []
    with open(mapdir2) as fin:
        for line in fin:
            pair = line.split()
            phonemap.append(pair[0])

    p = dict()
    for i in range(len(phonemap)):
        Feat = [0]*len(phonemap)
        Feat[i] = 1
        p[i] = np.asarray(Feat,dtype='float32').T

    return p

# For test.ark (without batch)
def rephraseOutput(oVector):
    dim = len(oVector[0])
    if dim == 39:
        temp = load_list39to48()
    elif dim == 1942:
        temp = load_liststateto48()

    output_label38 = []
    for sublist in oVector:
        # Due to the sigmoid property
        # pos = sublist.index(max(sublist))

        # Closest to Value
        Value = 1
        pos = min(enumerate(sublist), key=lambda x:abs(x[1]-Value))[0]

        output_label48.append(temp[pos])

    return output_label48

def load_list39to48():
    list39to48 = []
    with open(mapdir,'r') as fin:
        for line in fin:
            list39to48.append(line.split()[0])
    return list39to48

def load_liststateto48():
    liststateto48 = []
    with open(mapdir2,'r') as fin:
        for line in fin:
            liststateto48.append(line.split()[1])
    return liststateto48

def load_liststateto39():
    liststateto39 = []
    with open(mapdir2,'r') as fin:
        for line in fin:
            liststateto39.append(line.split()[0])
    return liststateto39

def load_dict_48to39():
    d = dict()
    with open(mapdir) as fin:
        for row in fin:
            pair = row.split()
            d[ pair[0] ] = pair[1]

    if len(set(d.keys())) != 48 or len(set(d.values())) != 39:
        print "Error phoneme amount!"

    return d

def load_dict_IdxPh48():
    d = dict()
    with open(mapdir2,'r') as fin:
        count = 0
        for line in fin:
            if count >= 48:
                break
            _, ph48, _ = line.split()
            d[ph48] = count
            count += 1
    return d

def load_list_ph48IdxtoChr():
    l = []
    with open(map48_chr) as fin:
        for line in fin:
            ph48, idx, _, char = line.split()
            l.append(char)
    return l

def load_dict_48toChr():
    d = dict()
    with open(map48_chr) as fin:
        for line in fin:
            ph48, idx, char, _ = line.split()
            d[ ph48 ] = char
    return d

def get_ph48_vec(i):
    v = np.zeros(48)
    v[i] = 1
    return v
