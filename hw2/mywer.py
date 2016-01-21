#!/usr/bin/env python
# from http://martin-thoma.com/word-error-rate-calculation/ 
import numpy
import sys

def wer(ref,hyp):
    d = numpy.zeros((len(ref)+1)*(len(hyp)+1), dtype=numpy.uint8)
    d = d.reshape((len(ref)+1, len(hyp)+1))
    for i in range(len(ref)+1):
        for j in range(len(hyp)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    #edit distance
    #print d[len(ref)][len(hyp)]

    #error rate
    return float(d[len(ref)][len(hyp)])#/float(len(r))

