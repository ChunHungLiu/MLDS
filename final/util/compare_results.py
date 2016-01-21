from collections import defaultdict
import csv
import sys

def main():
    csv1 = sys.argv[1]
    csv2 = sys.argv[2]

    f1 = open(csv1,'rb')
    f2 = open(csv2,'rb')

    reader1 = csv.reader(f1)
    reader2 = csv.reader(f2)

    next(reader1)
    next(reader2)

    dict1 = defaultdict(str)
    dict2 = defaultdict(str)

    for row1, row2 in zip(reader1,reader2):
        dict1[row1[0]] = row1[1]
        dict2[row2[0]] = row2[1]

    diff_count = float(len(set(dict1.iteritems())-set(dict2.iteritems())))
    total_count = float(len(set(dict1.iteritems())))

    print "{0} & {1} has {2} different entries, total entries {3}, about {4} different"\
            .format(csv1,csv2,diff_count, total_count, diff_count/total_count)


if __name__ == "__main__":
    main()
