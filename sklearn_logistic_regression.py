import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns


training_samples = pd.read_csv("train-set.csv", names=["ex1","ex2","ac"])
#print(training_samples.head(10))


#Original features
ex1 = training_samples[["ex1"]]
ex2 = training_samples[["ex2"]]

#Non linear features
ex3 = ex1.mul(ex2.values)
ex4 = ex1.mul(ex1.values)
ex5 = ex2.mul(ex2.values)

features = pd.concat([ex1,ex2,ex3,ex4,ex5], axis=1)
features.columns = ["x1","x2","x1_x2","x1_2","x2_2"]
labels = training_samples[["ac"]]
#print(features.head(10))


trained_results = pd.concat([ex1, ex2, pd.DataFrame(labels)], axis=1)
trained_results.columns = ['ex1','ex2','ac']
#sns.pairplot(x_vars=["ex1"], y_vars=["ex2"], data = trained_results,hue="ac", size=7)


model = linear_model.LogisticRegression()
model.fit(features, labels.values.ravel())


test_samples = pd.read_csv("test-set.csv",names=["ex1","ex2","ac"])
test_ex1 = test_samples[["ex1"]]
test_ex2 = test_samples[["ex2"]]
test_ex3 = test_ex1.mul(test_ex2.values)
test_ex4 = test_ex1.mul(test_ex1.values)
test_ex5 = test_ex2.mul(test_ex2.values)

test_features = pd.concat([test_ex1,test_ex2,test_ex3,test_ex4,test_ex5], axis=1)
test_features.columns = ["x1","x2","x1_x2","x1_2","x2_2"]

expected_labels = test_samples[["ac"]]

predicted_labels = model.predict(test_features)



test_results = pd.concat([test_ex1, test_ex2, pd.DataFrame(predicted_labels)], axis=1)
test_results.columns = ['ex1','ex2','ac']

sns.pairplot(x_vars=["ex1"], y_vars=["ex2"], data = test_results, hue="ac", size=7)

from sklearn import metrics
print(metrics.classification_report(expected_labels, predicted_labels, target_names = ["ACCEPTED", "REJECTED"]))
print("SCORE : {0}".format(metrics.accuracy_score(expected_labels, predicted_labels)))
