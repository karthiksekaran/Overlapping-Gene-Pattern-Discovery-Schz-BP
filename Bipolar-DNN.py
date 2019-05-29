import itertools
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn import metrics
from ggplot import *
import pandas as pd
from sklearn.model_selection import KFold
CANCER_TRAINING = "train.csv"
CANCER_TEST = "test.csv"
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=CANCER_TRAINING,
    target_dtype=np.int,
    features_dtype=np.int, KFold = 10)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=CANCER_TEST,
    target_dtype=np.int,
    features_dtype=np.int, KFold = 10)
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=60)]
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10,20, 10],
                                            n_classes=2,
                                            model_dir="DNN")
print(classifier)
prediction = classifier.fit(x=training_set.data, y=training_set.target, steps=2000).predict(test_set.data)
accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score*100))
expected  = test_set.target
predicted = list(prediction)
confusion_matrix(expected,predicted)
print(expected)
print(predicted)
print(classifier)
print(tf.estimator)
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix is Generated Successfully")
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cnf_matrix = confusion_matrix(expected, predicted)
np.set_printoptions(precision=2)
class_names = {"Control","Bipolar Disorder"}
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Normalized confusion matrix')
plt.show()
TP = 0
FP = 0
TN = 0
FN = 0

for i in range(len(predicted)):
    if expected[i]==predicted[i]==1:
       TP += 1
for i in range(len(predicted)):
    if np.all(predicted[i]==1) and np.all(expected[i]!=predicted[i]):
       FP += 1
for i in range(len(predicted)):
    if expected[i]==predicted[i]==0:
       TN += 1
for i in range(len(predicted)):
    if np.all(predicted[i]==0) and np.all(expected[i]!=predicted[i]):
       FN += 1
print ("True Positive:", TP)
print ("True Negative:", TN)
print ("False Positive:", FP)
print ("False Negative:", FN)

fpr, tpr, _ = metrics.roc_curve(expected, predicted)
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
disp = ggplot(df, aes(x='fpr', y='tpr')) +\
    geom_line() +\
    geom_abline(linetype='dashed')
print ("Executed Successfully")
print (disp)

print ("From SKLEARN")
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(expected, predicted)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print("True Positive:",TP)
print("True Negative:",TN)
print("False Positive:",FP)
print("False Negative:",FN)
print("Sensitivity:",100 * TP / (TP+FN))
print("Specificity:",100 * TN / (TN+FP))
print('Accuracy: {0:f}'.format(accuracy_score*100))
print("Precision:", TP/(TP+FP))
print("Recall:",TP/(TP+FN))
print("F-Score:", 2*TP/(2*TP+FP+FN))
res = "Accuracy of the Model is:" ,accuracy_score*100;