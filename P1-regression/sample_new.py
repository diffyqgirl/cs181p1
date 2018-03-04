
# coding: utf-8

# In[93]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn import linear_model
from rdkit import Chem
from rdkit.Chem import AllChem


# In[94]:

"""
Read in train and test as Pandas DataFrames
"""
print("here")
df_train = pd.read_csv("train_small.csv")
print("halfway")
df_test = pd.read_csv("test_small.csv",)
print("done")


# In[95]:

df_train.head()


# In[96]:

df_test.head()


# In[97]:

#store gap values
Y_train = df_train.gap.values
# Y_test = df_test.gap.values
X_train_smiles = df_train.smiles
X_test_smiles = df_test.smiles
#row where testing examples start
test_idx = df_train.shape[0]
#delete 'Id' column
# df_test = df_test.drop(['id'], axis=1)
#delete 'gap' column
# df_train = df_train.drop(['gap'], axis=1)


# In[98]:

#DataFrame with all train and test examples so we can more easily apply feature engineering on
df_all = pd.concat((df_train, df_test), axis=0)
df_all.head()


# In[99]:

"""
Example Feature Engineering

this calculates the length of each smile string and adds a feature column with those lengths
Note: this is NOT a good feature and will result in a lower score!
"""
#smiles_len = np.vstack(df_all.smiles.astype(str).apply(lambda x: len(x)))
#df_all['smiles_len'] = pd.DataFrame(smiles_len)


# In[100]:

#Drop the 'smiles' column
all_smiles = df_all.smiles
"""
df_all = df_all.drop(['smiles'], axis=1)
vals = df_all.values
X_train = vals[:test_idx]
X_test = vals[test_idx:]
print "Train features:", X_train.shape
print "Train gap:", Y_train.shape
print "Test features:", X_test.shape
"""


# In[101]:

# LR = LinearRegression()
# LR.fit(X_train, Y_train)
# LR_pred = LR.predict(X_test)


# In[102]:

"""
RF = RandomForestRegressor()
RF.fit(X_train, Y_train)
RF_pred = RF.predict(X_test)
"""


# In[103]:

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")


# In[104]:

# write_to_file("sample1.csv", LR_pred)
# write_to_file("sample2.csv", RF_pred)


# In[105]:

#mean_squared_error(LR_pred, Y_test)


# In[106]:

#mean_squared_error(RF_pred, Y_test)


# In[107]:

"""
X_scaled = preprocessing.scale(X_train)
RR = RidgeCV(alphas = np.linspace(0.01, 10, 100))
RR.fit(X_scaled, Y_train)
RR_pred = RR.predict(X_test)
print "mean squared error", mean_squared_error(RR_pred, Y_test)
print "RR.alpha", RR.alpha_
"""


# In[109]:
del df_train
del df_test
# morgans = list()
morgans = map(lambda x: list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x),2,nBits=1024)), all_smiles)
# print morgans[1]
del df_all
morgans_df = pd.DataFrame(data=morgans)
del morgans
print morgans_df.shape
print morgans_df.head()
morgans_vals = morgans_df.values
del morgans_df
X_train_morgans = morgans_vals[:test_idx]
X_test_morgans = morgans_vals[test_idx:]
LR = LinearRegression()
LR.fit(X_train_morgans, Y_train)
del X_train_morgans
del Y_train
LR_pred = LR.predict(X_test_morgans)
write_to_file("out.csv", LR_pred)
# print "mean squared error", mean_squared_error(LR_pred, Y_test)
# for i in range(df_all.shape()):
    # morgans.append(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles[i]),2,nBits=256)


# In[ ]:



