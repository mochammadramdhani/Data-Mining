import numpy as np
import matplotlib.pyplot as  plt
import pandas as pd 

# library
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


print('Data Kasus:')
print(x)
print(y)

# Menghilangkan Missing Value (nan)

imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print('Data Setelah Transformasi (Mean):')
print(x)

# Encoding Data Kategori (Atribut)

ct = ColumnTransformer (transformers=[('encoder', OneHotEncoder (), [0])], remainder= 'passthrough')
x = np. array(ct.fit_transform(x))

print ('Data Atribut:')
print(x)

# Encoding Data Kategori (Class/Label)
le = LabelEncoder ()
y = le.fit_transform (y)

print ('Data Kategori Class/Label:')
print(y)

# Membagi Dataset ke dalam Training Set dan Test Set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print ('Data Training Set dan Test Set:')
print('Data X Train:')
print(x_train)
print('Data Y Train:')
print(y_train)

print('Data X Test:')
print(x_test)
print('Data y Test:')
print(y_test)


sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print ('Data Feature Scaling:')
print('Data X Train:')
print(x_train)
print('Data X Test:')
print(x_test)