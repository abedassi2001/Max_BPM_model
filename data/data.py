import pandas as pd


data = pd.read_csv("gym_members_exercise_tracking.csv")

#print  the first row
print(data.head(0))

X = data.drop("Max_BPM", axis=1)
Y = data["Max_BPM"]

print(X.head(0))
print(Y.head(0))