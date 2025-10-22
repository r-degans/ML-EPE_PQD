import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import seaborn as sn
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_curve, precision_recall_curve, PrecisionRecallDisplay, roc_auc_score, RocCurveDisplay, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, f1_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn import svm

from datetime import datetime
import time
from concurrent.futures import ProcessPoolExecutor
output_dir = Path("confusion_matrix_outputs")
output_dir.mkdir(exist_ok=True)

cwd = Path.cwd()
noises = ["noiseless", "50db", "40db", "30db", "20db"]


def process(noise):
    print(noise)
    directory = cwd / str("Project_Data_EE4C12_EPE_PQD/SNR_" + noise)
    data = [pd.read_csv(f).drop(columns=['Unnamed: 0']) for f in directory.iterdir() if f.is_file()]

    files = [f.name for f in directory.iterdir()]
    # print(files)
    corrMatrices = [np.corrcoef(cat,rowvar=False) for cat in data]

    resultsFile = open(str("Task1-F1" + noise + ".txt"), 'w')

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
    # for i in range(8):
    #     print(files[i], end=' ')
    #     print(all_sorted_corrs[i])
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
        # print(keepList[:listLen])
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
        resultsFile.write('model1 = [')
        for file in range(9): # Step through all csv's (all PQD's)
            # print(files[file])
            Accuracy_LR = []
            Recall_LR = []
            F1_list = []
            modelList = []
            for dataDepth in range(1,9): # Step through all the feature sets
                dataX, dataY = dataGen(data, file, dataDepth)
            
                scaler = StandardScaler()
                scaler.fit(dataX)
                
                X_scaled = scaler.transform(dataX)
                
                Shuffle_state = 4720
                X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=Shuffle_state)
                X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.25, random_state=Shuffle_state) # 0.25 x 0.8 = 0.2

                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                X_val = scaler.transform(X_val)

                clf_lr = LogisticRegression(class_weight='balanced', tol=1e-6, max_iter=100_000).fit(X_train, y_train)
                modelList.append(clf_lr)
                y_prediction = clf_lr.predict(X_val)

                Accuracy_LR.append(accuracy_score(y_val, y_prediction)) 
                Recall_LR.append(recall_score(y_val, y_prediction))
                F1_list.append(f1_score(y_val, y_prediction))
            
            diff = (F1_list)
            # print(str("Linear " + files[file]))
            # print(modelList[np.argmax(F1_list)].feature_names_in_)
            # resultsFile.write(str("Linear model " + files[file] +  '\n'))
            resultsFile.write(str(max(F1_list)) + ',')
            # resultsFile.write(str(str((confusion_matrix(y_val, y_prediction).tolist())) + '\n'))
        resultsFile.write('] \n')
        # F1_LR = f1_score(y_test, y_prediction)
        # Precision_LR = precision_score(y_test, y_prediction)

        return Accuracy_LR, Recall_LR

    def linearLasso(data):
        resultsFile.write('model2 = [')
        param_grid = {
            'C': [0.01, 0.1, 1, 10],         # Regularization strength
            'penalty': ['l1', 'l2'],
            'solver' : ['liblinear']               # Type of regularization      # Solvers compatible with L1/L2
        }
        logreg = LogisticRegression(max_iter=100_000)
        grid_search = GridSearchCV(
            estimator=logreg,
            param_grid=param_grid,
            scoring='f1',  # simple F1 for binary classification
            cv=5,
            n_jobs=-1,
        )

        for file in range(9): # Step through all csv's (all PQD's)
            # print(files[file])
            Accuracy_LR = []
            Recall_LR = []
            modelList = []
            F1_list = []

            for dataDepth in range(1,9): # Step through all the feature sets
                dataX, dataY = dataGen(data, file, dataDepth)
            
                scaler = StandardScaler()
                scaler.fit(dataX)
                
                X_scaled = scaler.transform(dataX)
                
                Shuffle_state = 4720
                X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=Shuffle_state)
                X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.25, random_state=Shuffle_state) # 0.25 x 0.8 = 0.2

                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                X_val = scaler.transform(X_val)

                # clf_lr = LogisticRegression(class_weight='balanced', penalty='l1', solver='liblinear', C=40.0, max_iter=100_000, tol = 1e-6).fit(X_train, y_train)
                grid_search.fit(X_train, y_train)
                modelList.append(grid_search.best_estimator_)
                y_prediction = grid_search.best_estimator_.predict(X_val)

                Accuracy_LR.append(accuracy_score(y_val, y_prediction)) 
                Recall_LR.append(recall_score(y_val, y_prediction))
            
                F1_list.append(f1_score(y_val, y_prediction))
            
            resultsFile.write(str(max(F1_list)) + ',')
            # print(str("Linear lasso" + files[file]))
            # print(modelList[np.argmax(F1_list)].feature_names_in_)

            # resultsFile.write(str("linearLasso model " + files[file] +  '\n'))
            # resultsFile.write(str("F1 " + str(accuracy_score(y_val, y_prediction)) + \
            #      " Acc " + str(accuracy_score(y_val, y_prediction)) + ", Rec " + str(recall_score(y_val, y_prediction)) + '\n'))
            # resultsFile.write(str(str((confusion_matrix(y_val, y_prediction).tolist())) + '\n'))

        resultsFile.write('] \n')
        
        return Accuracy_LR, Recall_LR


    def SVM(data):
        resultsFile.write('model3 = [')
        
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf'],
            'gamma': ['scale', 'auto']  # only used for 'rbf' and 'poly'
        }
        svmInit = svm.SVC(class_weight='balanced', max_iter=100_000)
        grid_search = GridSearchCV(
            estimator=svmInit,
            param_grid=param_grid,
            scoring='f1',
            cv=4,       # 5-fold cross-validation
            n_jobs=-1   # use all available CPU cores
        )

        for file in range(9): # Step through all csv's (all PQD's)
            # print(files[file])
            Accuracy_LR = []
            Recall_LR = []
            modelList = []
            F1_list = []
            for dataDepth in range(1,9): # Step through all the feature sets
                dataX, dataY = dataGen(data, file, dataDepth)
            
                scaler = StandardScaler()
                scaler.fit(dataX)
                
                X_scaled = scaler.transform(dataX)
                
                Shuffle_state = 4720
                X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=Shuffle_state)
                X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.25, random_state=Shuffle_state) # 0.25 x 0.8 = 0.2

                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                X_val = scaler.transform(X_val)

                # clf_svmlin = svm.SVC(C=100.0, coef0=0.0, tol=1e-6, probability=True, class_weight='balanced').fit(X_train, y_train)
                grid_search.fit(X_train, y_train)
                modelList.append(grid_search.best_estimator_)
                y_prediction = grid_search.best_estimator_.predict(X_val)

                Accuracy_LR.append(accuracy_score(y_val, y_prediction)) 
                Recall_LR.append(recall_score(y_val, y_prediction))
                F1_list.append(f1_score(y_val, y_prediction))

            resultsFile.write(str(max(F1_list)) + ',')
            # print(str("SVM " + files[file]))

            # print(modelList[np.argmax(F1_list)].feature_names_in_)


            # resultsFile.write(str("SVM model " + files[file] +  '\n'))
            # resultsFile.write(str("F1 " + str(accuracy_score(y_val, y_prediction)) + \
            #      " Acc " + str(accuracy_score(y_val, y_prediction)) + ", Rec " + str(recall_score(y_val, y_prediction)) + '\n'))
            # resultsFile.write(str(str((confusion_matrix(y_val, y_prediction).tolist())) + '\n'))
        resultsFile.write('] \n')

        return Accuracy_LR, Recall_LR

    linearModel(data)
    linearLasso(data)
    SVM(data)

start = time.time()
with ProcessPoolExecutor() as executor:
    results = list(executor.map(process, noises))

print(time.time() - start)
