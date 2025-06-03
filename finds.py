import pandas as pd
def find_s_algorithm(data):
    features = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values
    hypothesis = None
    for i in range(len(features)):
        if target[i] == 'Yes':
            hypothesis = list(features[i])
            break
    if hypothesis is None:
        print("No positive examples found in the data.")
        return None
    for i in range(len(features)):
        if target[i] == 'Yes':
            for j in range(len(hypothesis)):
                if hypothesis[j] != features[i][j]:
                    hypothesis[j] = '?'
    return hypothesis
data = pd.read_csv(r'C:\Users\shara\Desktop\Lab4.csv')
print("Training Data:")
print(data)
hypothesis = find_s_algorithm(data)
if hypothesis is not None:
    print("\nThe most specific hypothesis consistent with the training examples is:")
    print(hypothesis)
