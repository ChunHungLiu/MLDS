def get_phone2index_dict(mapfile, phone = 48, reverse = False):
    phonemap = []
    with open(mapfile) as fin:
        for line in fin:
            pair = line.split()
            if phone == 48:
                phonemap.append(pair[0])
            elif phone == 39:
                phonemap.append(pair[1])

    phonemap = list(set(phonemap))

    p = dict()
    for i in range(len(phonemap)):
        p[phonemap[i]] = i

    if reverse:
        return { v:k for k,v in p.items() }
    return p

def get_ID2index_dict(ID2phone_dict,phone2index_dict):
    ret = dict()
    for key in ID2phone_dict.keys():
        ret[key] = phone2index_dict[ID2phone_dict[key]]
    return ret

def get_phone49to39_dict(mapfile):
    ret = dict()
    with open(mapdir) as fin:
        for row in fin:
            words = row.strip('\n').split()
            ret[words[0]] = words[1]
    return ret
