import pandas as pd
from pathlib import Path
import numpy as np
cwd = Path.cwd()
directory = cwd / "Project_Data_EE4C12_EPE_PQD/SNR_noiseless"

data = [pd.read_csv(f) for f in directory.iterdir() if f.is_file()]
files = [f.name for f in directory.iterdir()]
print(files)

######## 

def featureSnipper(data, sortedArray, listDepth):
    topList = sortedArray.head(listDepth)
    keepList = list(pd.unique(pd.concat([topList['Var1'], topList['Var2']])))
    return data[keepList]

    