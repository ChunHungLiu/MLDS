def loss(y, y_bar):
    d = np.zeros((len(y)+1,len(y_bar)+1))
    for i in range(len(y)+1):
        for j in range(len(y_bar)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    # computation
    for i in range(1, len(y)+1):
        for j in range(1, len(y_bar)+1):
            if y[i-1] == y_bar[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    #edit distance
    print float(d[len(y)][len(y_bar)])/float(len(y))
    return float(d[len(y)][len(y_bar)])