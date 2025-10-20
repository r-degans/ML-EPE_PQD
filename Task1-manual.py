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

def save_confusion_matrix_plot(folder, plot_title="confusion_matrix"):
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    filename = f"{plot_title}_{timestamp}.png"
    filepath = folder / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath

cwd = Path.cwd()
directory = cwd / "Project_Data_EE4C12_EPE_PQD/SNR_50db"
noise = '50dB'
files = [f.name for f in directory.iterdir()]
data = [pd.read_csv(f).drop(columns=['Unnamed: 0']) for f in directory.iterdir() if f.is_file()]

resultsFile = open(str("Task1-manual-" + noise + ".txt"), 'w')


# Example: dictionary mapping each file to its list of features to keep
selected_features_per_file = {
    "Swell.csv": ["sd", "max", "avg abs"],
    "Transient.csv": ["sd", "max", "sd fma", "no. pt near 0"],
    "Swell_harmonics.csv": ["sd", "max", "no. pt near 0", "avg abs"],
    "Normal.csv": ["min", "max", "sd"],
    "Sag_harmonics.csv": ["no. pt near 0", "max", "sd", "avg abs"],
    "Sag.csv": ["no. pt near 0", "max", "sd", "avg abs"],
    "Interruption.csv": ["no. pt near 0", "max", "sd", "avg abs"],
    "Harmonics.csv": ["no. pt near 0", "min", "max", "sd"],
    "Flicker.csv": ["min", "max", "sd", "no. pt near 0"]
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

def dataGen(data, file, dataDepth):
    allData = pd.concat([data[file]] + data[:file] + data[file+1:])
    # allData = featureSnipper(allData, all_sorted_corrs[file], dataDepth)
    
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