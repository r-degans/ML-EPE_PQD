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
from sklearn.metrics import f1_score, accuracy_score, roc_curve, precision_recall_curve, PrecisionRecallDisplay, roc_auc_score, RocCurveDisplay, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
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
    directory = cwd / str("Project_Data_EE4C12_EPE_PQD/SNR_" + noise)
    files = [f.name for f in directory.iterdir()]
    data = [pd.read_csv(f).drop(columns=['Unnamed: 0']) for f in directory.iterdir() if f.is_file()]

    resultsFile = open(str("Task1-manual-F1" + noise + ".txt"), 'w')

    # Example: dictionary mapping each file to its list of features to keep
    selected_features_per_file = {
        "Swell.csv": ["no. pt near 0", "sd", "max", "avg abs"],
        "Transient.csv": ["no. pt near 0", "sd", "max", "sd fma", "no. pks"],
        "Swell_harmonics.csv": ["no. pt near 0", "sd fma 3", "max", "sd", "avg abs"],
        "Normal.csv": ["min", "max", "sd"],
        "Sag_harmonics.csv": ["no. pt near 0","sd", "max", "sd fma 3", "avg abs"],
        "Sag.csv": ["no. pt near 0", "max", "sd", "avg abs"],
        "Interruption.csv": ["no. pt near 0","sd", "avg abs", "max"],
        "Harmonics.csv": ["no. pt near 0", "min", "sd fma 3"],
        "Flicker.csv": ["min", "no. pt near 0","sd", "max"]
        # Add entries for all files
    }
    filtered_dfs = []

    for file in directory.iterdir():
        if file.suffix.lower() == ".csv":
            df = pd.read_csv(file)

            # Drop first column (index)
            df = df.iloc[:, 1:]

            # Get the list of features to keep for this file
            features_to_keep = selected_features_per_file.get(file.name, [])

            # Keep only the features that exist in the file
            df_filtered = df.loc[:, [f for f in features_to_keep if f in df.columns]]

            filtered_dfs.append(df_filtered)

            print(f"{file.name}: kept {len(df_filtered.columns)} features")

    # Optional: combine all filtered DataFrames
    combined_df = pd.concat(filtered_dfs, ignore_index=True)
    print("\nCombined filtered data shape:", combined_df.shape)

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
    bigData = combined_df

    def linearModel(data):
        resultsFile.write('model4 = [')
        for file in range(9): # Step through all csv's (all PQD's)
            print(files[file])
            Accuracy_LR = []
            Recall_LR = []
            modelList = []

            dataX, dataY = dataGen(data, file)
        
            scaler = StandardScaler()
            scaler.fit(dataX)
            
            X_scaled = scaler.transform(dataX)
            
            Shuffle_state = 4720
            X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=Shuffle_state)
            X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.25, random_state=Shuffle_state) # 0.25 x 0.8 = 0.2


            clf_lr = LogisticRegression(class_weight='balanced', tol=1e-6, max_iter=10_000).fit(X_train, y_train)
            y_prediction = clf_lr.predict(X_val)

            Accuracy_LR.append(accuracy_score(y_val, y_prediction)) 
            Recall_LR.append(recall_score(y_val, y_prediction))
            
            resultsFile.write(str(f1_score(y_val, y_prediction)) + ',')
            
            # resultsFile.write(str("Linear model " + files[file] +  '\n'))
            # resultsFile.write(str("F1 " + str(accuracy_score(y_val, y_prediction)) + \
            #     " Acc " + str(accuracy_score(y_val, y_prediction)) + ", Rec " + str(recall_score(y_val, y_prediction)) + '\n'))
            # resultsFile.write(str(str((confusion_matrix(y_val, y_prediction).tolist())) + '\n'))

        # F1_LR = f1_score(y_test, y_prediction)
        # Precision_LR = precision_score(y_test, y_prediction)
        resultsFile.write('] \n')

        return Accuracy_LR, Recall_LR

    def linearLasso(data):
        resultsFile.write('model5 = [')
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
            print(files[file])
            Accuracy_LR = []
            Recall_LR = []
            modelList = []

            dataX, dataY = dataGen(data, file)
        
            scaler = StandardScaler()
            scaler.fit(dataX)
            
            X_scaled = scaler.transform(dataX)
            
            Shuffle_state = 4720
            X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=Shuffle_state)
            X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.25, random_state=Shuffle_state) # 0.25 x 0.8 = 0.2

            # clf_lr = LogisticRegression(class_weight='balanced', penalty='l1', solver='liblinear', C=40.0, max_iter=10_000, tol = 1e-6).fit(X_train, y_train)
            # modelList.append(clf_lr)
            # y_prediction = clf_lr.predict(X_test)
            grid_search.fit(X_train, y_train)
            modelList.append(grid_search.best_estimator_)
            y_prediction = grid_search.best_estimator_.predict(X_val)

            resultsFile.write(str(f1_score(y_val, y_prediction)) + ',')
            
            # resultsFile.write(str("linearLasso model " + files[file] +  '\n'))
            # resultsFile.write(str("F1 " + str(accuracy_score(y_val, y_prediction)) + \
            #      " Acc " + str(accuracy_score(y_val, y_prediction)) + ", Rec " + str(recall_score(y_val, y_prediction)) + '\n'))
            # resultsFile.write(str(str((confusion_matrix(y_val, y_prediction).tolist())) + '\n'))


        resultsFile.write('] \n')
        
        return Accuracy_LR, Recall_LR


    def SVM(data):
        
        resultsFile.write('model6 = [')
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
            print(files[file])

            Accuracy_LR = []
            Recall_LR = []
            modelList = []

            dataX, dataY = dataGen(data, file)
        
            scaler = StandardScaler()
            scaler.fit(dataX)
            
            X_scaled = scaler.transform(dataX)
            
            Shuffle_state = 4720
            X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=Shuffle_state)
            X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.25, random_state=Shuffle_state) # 0.25 x 0.8 = 0.2

            grid_search.fit(X_train, y_train)
            y_prediction = grid_search.best_estimator_.predict(X_val)

            # clf_svmlin = svm.SVC(C=100.0, coef0=0.0, tol=1e-6, probability=True, class_weight='balanced').fit(X_train, y_train)
            # modelList.append(clf_svmlin)
            # y_prediction= clf_svmlin.predict(X_test)

            resultsFile.write(str(f1_score(y_val, y_prediction)) + ',')
            
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
