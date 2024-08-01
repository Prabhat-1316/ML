import csv
with open('tennis.csv','r') as file:
    reader = csv.reader(file)
    data = list(reader)
'''
# Exp-1:FIND-S Algorithm
def FIND_S(data):
    hypothesis = None
    for ex in data:
        print(ex)
        if ex[-1] == 'True':
            if hypothesis is None:
                hypothesis = ex[:-1]
            else:
                for i in range(len(hypothesis)):
                    if hypothesis[i] != ex[i]:
                        hypothesis[i] = '?'
    return hypothesis

output = FIND_S(data)
print(output)
'''
#Exp-2: Candidate Algorithm



