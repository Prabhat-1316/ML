'''exp-1:For a given set of training data examples stored in a .CSV file, implement and
demonstrate the Find-S algorithm to output a description of the set of all
hypotheses consistent with the training examples
'''
import csv

# Function to read data from a CSV file
def read_data_from_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

# Find-S algorithm implementation
def find_s(data):
    
    # Initialize the most specific hypothesis with the first positive example
    hypothesis = None

    for instance in data:
        print(instance)
        if instance[-1] == 'True':  # Check if the instance is positive
            if hypothesis is None:
                hypothesis = instance[:-1]  # Initialize the hypothesis
                
            else:
                for i in range(len(hypothesis)):
                    if hypothesis[i] != instance[i]:
                        hypothesis[i] = '?'  # Generalize the hypothesis
                        
    
    return hypothesis

if __name__ == "__main__":
    # Read the dataset from the CSV file
    data = read_data_from_csv('tennis.csv')
    
    # Apply the Find-S algorithm
    hypothesis = find_s(data)
    
    # Print the most specific hypothesis
    print("Most specific hypothesis:", hypothesis)
'''OUTPUT
["'Sunny'", " 'Warm'", " 'Normal'", " 'Strong'", " 'Warm'", " 'Same'", 'True']
["'Sunny'", " 'Warm'", " 'High'", " 'Strong'", " 'Warm'", "'Same'", 'True']
["'Rainy'", " 'Cold'", " 'High'", " 'Strong'", " 'Warm'", "'Change'", 'False']
["'Sunny'", " 'Warm'", " 'High'", " 'Strong'", " 'Cool'", "'Change'", 'True']
Most specific hypothesis: ["'Sunny'", " 'Warm'", '?', " 'Strong'", '?', '?']

'''