import csv
import os

predictionsdir = './results/prediction'

with open('test_solution.csv') as solfile:
    sol_csv = csv.reader(solfile)
    next(sol_csv,None)
    sol = set([ (row[0],row[1]) for row in sol_csv  ])

for accfilename in os.listdir(predictionsdir):
    if accfilename.endswith('.csv'):
        with open(predictionsdir+accfilename) as accfile:
            acc_csv = csv.reader(accfile)
            next(acc_csv,None)
            acc = set([ (row[0],row[1]) for row in acc_csv ])
            total = len(acc)
            print accfilename+":",float(len(acc & sol))/total
