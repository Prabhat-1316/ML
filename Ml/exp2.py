'''exp-2:For a given set of training data examples stored in a .CSV file, implement and
demonstrate the Candidate-Elimination algorithm to output a description of
the set of all hypotheses consistent with the training examples
'''
import csv
with open('tennis.csv','r') as file:
    reader = csv.reader(file)
    data = list(reader)

def candidate_elimination(data):
    # Initialize the most specific hypothesis s
    s = data[0][:-1]  # The first example (excluding the label)
    # Initialize the most general hypothesis g
    g = [['?' for i in range(len(s))] for j in range(len(s))]
    
    for instance in data:
        if instance[-1]=='True':  # If the instance is positive
            for j in range(len(s)):
                if instance[j] != s[j]:
                    s[j] = '?'
                    g[j][j] = '?'
        else:  # If the instance is negative
            for j in range(len(s)):
                if instance[j] != s[j]:
                    g[j][j] = s[j]
                else:
                    g[j][j] = "?"
        
        print(f"\nSteps of Candidate Elimination Algorithm")
        print("Specific hypothesis:", s)
        print("General hypothesis:", g)
    
    gh = []
    for i in g:
        if '?' not in i:
            gh.append(i)
    
    print("\nFinal specific hypothesis:\n", s)
    print("\nFinal general hypothesis:\n", gh)

if __name__ == "__main__":
    candidate_elimination(data)

'''output
Steps of Candidate Elimination Algorithm
Specific hypothesis: ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same']
General hypothesis: [['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]

Steps of Candidate Elimination Algorithm
Specific hypothesis: ['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']
General hypothesis: [['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]

Steps of Candidate Elimination Algorithm
Specific hypothesis: ['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']
General hypothesis: [['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', 'Same']]

Steps of Candidate Elimination Algorithm
Specific hypothesis: ['Sunny', 'Warm', '?', 'Strong', '?', '?']
General hypothesis: [['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]   

Final specific hypothesis:
 ['Sunny', 'Warm', '?', 'Strong', '?', '?']

Final general hypothesis:
 []
 '''