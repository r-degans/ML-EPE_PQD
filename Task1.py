import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import seaborn as sn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_curve, precision_recall_curve, PrecisionRecallDisplay, roc_auc_score, RocCurveDisplay, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn import svm

from datetime import datetime

output_dir = Path("confusion_matrix_outputs")
output_dir.mkdir(exist_ok=True)



cwd = Path.cwd()
directory = cwd / "Project_Data_EE4C12_EPE_PQD/SNR_50db"
noise = "50dB"
data = [pd.read_csv(f).drop(columns=['Unnamed: 0']) for f in directory.iterdir() if f.is_file()]

files = [f.name for f in directory.iterdir()]
print(files)
corrMatrices = [np.corrcoef(cat,rowvar=False) for cat in data]

resultsFile = open(str("Task1-" + noise + ".txt"), 'w')

# i = 0
# for e in corrMatrices:
#     sn.heatmap(e, annot=True)
#     plt.title(files[i])
#     plt.show()
#     i = i + 1


## Sort features by correlation 

colnames = data[0].columns

all_sorted_corrs = []

for i, corr_matrix in enumerate(corrMatrices):
    df = pd.DataFrame(corr_matrix, index=colnames, columns=colnames)

    # Take only upper triangle (to avoid duplicates and self-correlations)
    corr_pairs = (
        df.where(np.triu(np.ones(df.shape), k=1).astype(bool))
        .stack()
        .reset_index()
        .rename(columns={'level_0': 'Var1', 'level_1': 'Var2', 0: 'Correlation'})
    )

    # Add absolute value and sort
    corr_pairs['AbsCorr'] = corr_pairs['Correlation'].abs()
    corr_pairs_sorted = (
        corr_pairs.sort_values('AbsCorr', ascending=False)
                  .drop(columns='AbsCorr')
                  .reset_index(drop=True)
    )

    all_sorted_corrs.append(corr_pairs_sorted)
for i in range(8):
    print(files[i], end=' ')
    print(all_sorted_corrs[i])
# def featureSnipper(data, sortedArray, listDepth):
#     topList = sortedArray.head(listDepth)
#     keepList = list(pd.unique(pd.concat([topList['Var1'], topList['Var2']])))
#     return data[keepList]

def featureSnipper(data, sortedArray, listLen):
    Len = 0
    listDepth = 1
    while Len < listLen:
        topList = sortedArray.head(listDepth)
        keepList = list(pd.unique(pd.concat([topList['Var1'], topList['Var2']])))
        Len = len(keepList)
        listDepth = listDepth + 1
        if(listDepth > 35):
            return data[keepList[:listLen]]
    print(keepList[:listLen])
    return data[keepList[:listLen]]

Accuracy_LR = []
Recall_LR = []
bigData = pd.concat(data)

def dataGen(data, file, dataDepth):
    allData = pd.concat([data[file]] + data[:file] + data[file+1:])
    allData = featureSnipper(allData, all_sorted_corrs[file], dataDepth)
    
    y = [1]*149
    y.extend([0]*(8*149))
    
    return allData, y

def linearModel(data):
    for file in range(9): # Step through all csv's (all PQD's)
        print(files[file])
        Accuracy_LR = []
        Recall_LR = []
        modelList = []
        for dataDepth in range(1,9): # Step through all the feature sets
            dataX, dataY = dataGen(data, file, dataDepth)
        
            scaler = StandardScaler()
            scaler.fit(dataX)
            
            X_scaled = scaler.transform(dataX)
            
            Shuffle_state = 4720
            X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=Shuffle_state)
            X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.25, random_state=Shuffle_state) # 0.25 x 0.8 = 0.2

            clf_lr = LogisticRegression(class_weight='balanced', tol=1e-6, max_iter=10_000).fit(X_train, y_train)
            modelList.append(clf_lr)
            y_prediction = clf_lr.predict(X_test)

            Accuracy_LR.append(accuracy_score(y_test, y_prediction)) 
            Recall_LR.append(recall_score(y_test, y_prediction))
        
        diff = np.array(Accuracy_LR) + np.array(Recall_LR)
        bestModel = modelList[np.argmax(diff)]

        y_prediction = clf_lr.predict(X_val)
        resultsFile.write(str("Linear model " + files[file] +  '\n'))
        resultsFile.write(str(str((confusion_matrix(y_val, y_prediction).tolist())) + '\n'))

    # F1_LR = f1_score(y_test, y_prediction)
    # Precision_LR = precision_score(y_test, y_prediction)

    return Accuracy_LR, Recall_LR

def linearLasso(data):
    for file in range(9): # Step through all csv's (all PQD's)
        print(files[file])
        Accuracy_LR = []
        Recall_LR = []
        modelList = []
        for dataDepth in range(1,9): # Step through all the feature sets
            dataX, dataY = dataGen(data, file, dataDepth)
        
            scaler = StandardScaler()
            scaler.fit(dataX)
            
            X_scaled = scaler.transform(dataX)
            
            Shuffle_state = 4720
            X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=Shuffle_state)
            X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.25, random_state=Shuffle_state) # 0.25 x 0.8 = 0.2

            clf_lr = LogisticRegression(class_weight='balanced', penalty='l1', solver='liblinear', C=40.0, max_iter=10_000, tol = 1e-6).fit(X_train, y_train)
            modelList.append(clf_lr)
            y_prediction = clf_lr.predict(X_test)

            Accuracy_LR.append(accuracy_score(y_test, y_prediction)) 
            Recall_LR.append(recall_score(y_test, y_prediction))
        
        diff = np.array(Accuracy_LR) + np.array(Recall_LR)
        bestModel = modelList[np.argmax(diff)]

        y_prediction = clf_lr.predict(X_val)

        resultsFile.write(str("linearLasso model " + files[file] +  '\n'))
        resultsFile.write(str(str((confusion_matrix(y_val, y_prediction).tolist())) + '\n'))

    
    return Accuracy_LR, Recall_LR


def SVM(data):
    
    for file in range(9): # Step through all csv's (all PQD's)
        print(files[file])

        Accuracy_LR = []
        Recall_LR = []
        modelList = []
        for dataDepth in range(1,9): # Step through all the feature sets
            dataX, dataY = dataGen(data, file, dataDepth)
        
            scaler = StandardScaler()
            scaler.fit(dataX)
            
            X_scaled = scaler.transform(dataX)
            
            Shuffle_state = 4720
            X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=Shuffle_state)
            X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.25, random_state=Shuffle_state) # 0.25 x 0.8 = 0.2

            clf_svmlin = svm.SVC(C=10.0, coef0=0.0, tol=1e-6, probability=True, class_weight={0:0.545, 1:6}).fit(X_train, y_train)
            modelList.append(clf_svmlin)
            y_prediction= clf_svmlin.predict(X_test)

            Accuracy_LR.append(accuracy_score(y_test, y_prediction)) 
            Recall_LR.append(recall_score(y_test, y_prediction))
        
        diff = np.array(Accuracy_LR) + np.array(Recall_LR)
        bestModel = modelList[np.argmax(diff)]

        y_prediction = clf_svmlin.predict(X_val)
        resultsFile.write(str("SVM model " + files[file] +  '\n'))
        resultsFile.write(str(str((confusion_matrix(y_val, y_prediction).tolist())) + '\n'))

    return Accuracy_LR, Recall_LR

linearModel(data)
linearLasso(data)
SVM(data)