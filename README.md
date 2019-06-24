# Data_Science_challenge


```python
#data analysis and wrangling
import numpy as np
import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
```


```python
# Loading train and test data
df_train= pd.read_csv('C:/Users/Mayank/Downloads/4eddd640-9-dataset/dataset/train.csv')
df_test=pd.read_csv('C:/Users/Mayank/Downloads/4eddd640-9-dataset/dataset/test.csv')
```


```python
#checking train data
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Inv_Id</th>
      <th>Vendor_Code</th>
      <th>GL_Code</th>
      <th>Inv_Amt</th>
      <th>Item_Description</th>
      <th>Product_Category</th>
      <th>item_description1</th>
      <th>item_description2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>VENDOR-61</td>
      <td>GL-6050100</td>
      <td>6.973473</td>
      <td>AETNA VARIABLE FUND - Apr-2002 - Store Managem...</td>
      <td>CLASS-784</td>
      <td>AETNA VARIABLE FUND</td>
      <td>Store Management Real Estate Real Estate Serv...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>VENDOR-61</td>
      <td>GL-6050100</td>
      <td>25.053841</td>
      <td>AETNA VARIABLE FUND - Nov-2000 - Store Managem...</td>
      <td>CLASS-784</td>
      <td>AETNA VARIABLE FUND</td>
      <td>Store Management Real Estate Real Estate Serv...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>VENDOR-449</td>
      <td>GL-6050100</td>
      <td>53.573737</td>
      <td>FAIRCHILD CORP - Nov-2001 - Store Management R...</td>
      <td>CLASS-784</td>
      <td>AETNA VARIABLE FUND</td>
      <td>Store Management Real Estate Real Estate Serv...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>VENDOR-682</td>
      <td>GL-6050100</td>
      <td>67.388827</td>
      <td>CALIFORNIA REAL ESTATE INVESTMENT TRUST - Aug-...</td>
      <td>CLASS-784</td>
      <td>AETNA VARIABLE FUND</td>
      <td>Store Management Real Estate Real Estate Serv...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>VENDOR-682</td>
      <td>GL-6050100</td>
      <td>74.262047</td>
      <td>CALIFORNIA REAL ESTATE INVESTMENT TRUST - Mar-...</td>
      <td>CLASS-784</td>
      <td>AETNA VARIABLE FUND</td>
      <td>Store Management Real Estate Real Estate Serv...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#checking test data
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Inv_Id</th>
      <th>Vendor_Code</th>
      <th>GL_Code</th>
      <th>Inv_Amt</th>
      <th>Item_Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>VENDOR-1197</td>
      <td>GL-6050100</td>
      <td>10.916343</td>
      <td>DESOTO INC - Jul-2008 - Store Management Real ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12</td>
      <td>VENDOR-792</td>
      <td>GL-6050100</td>
      <td>38.658772</td>
      <td>CENTURY REALTY TRUST - Nov-2019 - Store Manage...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>VENDOR-792</td>
      <td>GL-6050100</td>
      <td>46.780476</td>
      <td>CENTURY REALTY TRUST - Jan-2006 - Store Manage...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18</td>
      <td>VENDOR-792</td>
      <td>GL-6050100</td>
      <td>7.058866</td>
      <td>CENTURY REALTY TRUST - Sep-2002 - Store Manage...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>VENDOR-792</td>
      <td>GL-6050100</td>
      <td>32.931765</td>
      <td>CENTURY REALTY TRUST - Nov-2018 - Store Manage...</td>
    </tr>
  </tbody>
</table>
</div>



We have to predict the <b> Product Category</b> 

We will look at different features. First let us look at different <b> product categories</b>


```python
df_train['Product_Category'].unique()
```




    array(['CLASS-784', 'CLASS-489', 'CLASS-913', 'CLASS-368', 'CLASS-816',
           'CLASS-629', 'CLASS-177', 'CLASS-123', 'CLASS-671', 'CLASS-804',
           'CLASS-453', 'CLASS-1042', 'CLASS-95', 'CLASS-49', 'CLASS-947',
           'CLASS-110', 'CLASS-278', 'CLASS-522', 'CLASS-606', 'CLASS-651',
           'CLASS-765', 'CLASS-953', 'CLASS-839', 'CLASS-668', 'CLASS-758',
           'CLASS-942', 'CLASS-764', 'CLASS-50', 'CLASS-51', 'CLASS-559',
           'CLASS-75', 'CLASS-74', 'CLASS-783', 'CLASS-323', 'CLASS-322',
           'CLASS-720', 'CLASS-230', 'CLASS-571'], dtype=object)




```python
len(df_train['Product_Category'].unique())
```




    38



There are 38 categories of products.<br>
Let us analyze Item_description. It looks it has more information.


```python
len(df_train['Item_Description'].unique()) #Thats a lot of unique values
```




    5118




```python
df_train['Item_Description'].head() # We can break this and introduce more parameters for prediction
```




    0    AETNA VARIABLE FUND - Apr-2002 - Store Managem...
    1    AETNA VARIABLE FUND - Nov-2000 - Store Managem...
    2    FAIRCHILD CORP - Nov-2001 - Store Management R...
    3    CALIFORNIA REAL ESTATE INVESTMENT TRUST - Aug-...
    4    CALIFORNIA REAL ESTATE INVESTMENT TRUST - Mar-...
    Name: Item_Description, dtype: object




```python
df_train['Item_Description'][1].split("-")
```




    ['AETNA VARIABLE FUND ',
     ' Nov',
     '2000 ',
     ' Store Management Real Estate Real Estate Services Real Estate General (Search, Appraisal, Realtor Commission)']




```python
#Let us create two new columns item_description1 and item_description2 from information from item_description
df_train['item_description1']=df_train['Item_Description'][1].split("-")[0]
df_train['item_description2']=df_train['Item_Description'][1].split("-")[3]
```


```python
#set each column
for i in range(0,len(df_train)):
    df_train['item_description1'][i] = df_train['Item_Description'][i].split("-")[0]
```

    C:\Users\Mayank\Anaconda3\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    


```python
for i in range(0,len(df_train)):
    df_train['item_description2'][i] = df_train['Item_Description'][i].split("-")[3]
```

    C:\Users\Mayank\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    


```python
df_train.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Inv_Id</th>
      <th>Vendor_Code</th>
      <th>GL_Code</th>
      <th>Inv_Amt</th>
      <th>Item_Description</th>
      <th>Product_Category</th>
      <th>item_description1</th>
      <th>item_description2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5714</th>
      <td>8007</td>
      <td>VENDOR-401</td>
      <td>GL-6121905</td>
      <td>89.409831</td>
      <td>BAGDAD CHASE INC - Jul-2001 - Printed Collater...</td>
      <td>CLASS-571</td>
      <td>BAGDAD CHASE INC</td>
      <td>Printed Collateral Miscellaneous Printed Mate...</td>
    </tr>
    <tr>
      <th>5715</th>
      <td>8008</td>
      <td>VENDOR-401</td>
      <td>GL-6121905</td>
      <td>35.066517</td>
      <td>BAGDAD CHASE INC - May-2003 - Printed Collater...</td>
      <td>CLASS-571</td>
      <td>BAGDAD CHASE INC</td>
      <td>Printed Collateral Miscellaneous Printed Mate...</td>
    </tr>
    <tr>
      <th>5716</th>
      <td>8009</td>
      <td>VENDOR-1550</td>
      <td>GL-6121905</td>
      <td>51.270765</td>
      <td>FIFTH AVENUE SECURITIES CORP - Mar-2019 - Prin...</td>
      <td>CLASS-571</td>
      <td>FIFTH AVENUE SECURITIES CORP</td>
      <td>Printed Collateral Miscellaneous Printed Mate...</td>
    </tr>
    <tr>
      <th>5717</th>
      <td>8011</td>
      <td>VENDOR-698</td>
      <td>GL-6121905</td>
      <td>42.693898</td>
      <td>CANADA DRY BOTTLING CO OF FLORIDA INC - Mar-20...</td>
      <td>CLASS-571</td>
      <td>CANADA DRY BOTTLING CO OF FLORIDA INC</td>
      <td>Printed Collateral Miscellaneous Printed Mate...</td>
    </tr>
    <tr>
      <th>5718</th>
      <td>8012</td>
      <td>VENDOR-698</td>
      <td>GL-6121905</td>
      <td>99.841762</td>
      <td>CANADA DRY BOTTLING CO OF FLORIDA INC - Jun-20...</td>
      <td>CLASS-571</td>
      <td>CANADA DRY BOTTLING CO OF FLORIDA INC</td>
      <td>Printed Collateral Miscellaneous Printed Mate...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#unique values item description1 and description2
des1_unique=df_train['item_description1'].unique()
des2_unique=df_train['item_description2'].unique()
```


```python
len_des1=len(des1_unique)
len_des2=len(des2_unique)
```


```python
gl_code_unique=df_train['GL_Code'].unique()
```


```python
vendor_unique=df_train['Vendor_Code'].unique()
```


```python
sns.boxplot(x='Product_Category', y='Inv_Amt', data=df_train,width=0.5)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x23f8c020208>




![png](output_20_1.png)


We need to convert categorical values to float values to make data ready for prediction


```python
df_train['GL_Code']=df_train['GL_Code'].replace(gl_code_unique[0],0)
df_train['GL_Code']=df_train['GL_Code'].replace(gl_code_unique[1],1)
df_train['GL_Code']=df_train['GL_Code'].replace(gl_code_unique[2],2)
df_train['GL_Code']=df_train['GL_Code'].replace(gl_code_unique[3],3)
df_train['GL_Code']=df_train['GL_Code'].replace(gl_code_unique[4],4)
df_train['GL_Code']=df_train['GL_Code'].replace(gl_code_unique[5],5)
df_train['GL_Code']=df_train['GL_Code'].replace(gl_code_unique[6],6)
df_train['GL_Code']=df_train['GL_Code'].replace(gl_code_unique[7],7)
df_train['GL_Code']=df_train['GL_Code'].replace(gl_code_unique[8],8)

```


```python
for i in range(len(vendor_unique)):
    df_train['Vendor_Code']=df_train['Vendor_Code'].replace(vendor_unique[i],i)
for j in range(len(des2_unique)):
    df_train['item_description2']=df_train['item_description2'].replace(des2_unique[j],j)
for k in range(len(des1_unique)):
    df_train['item_description1']=df_train['item_description1'].replace(des1_unique[k],k)
```


```python
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Inv_Id</th>
      <th>Vendor_Code</th>
      <th>GL_Code</th>
      <th>Inv_Amt</th>
      <th>Item_Description</th>
      <th>Product_Category</th>
      <th>item_description1</th>
      <th>item_description2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6.973473</td>
      <td>AETNA VARIABLE FUND - Apr-2002 - Store Managem...</td>
      <td>CLASS-784</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>25.053841</td>
      <td>AETNA VARIABLE FUND - Nov-2000 - Store Managem...</td>
      <td>CLASS-784</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>53.573737</td>
      <td>FAIRCHILD CORP - Nov-2001 - Store Management R...</td>
      <td>CLASS-784</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>67.388827</td>
      <td>CALIFORNIA REAL ESTATE INVESTMENT TRUST - Aug-...</td>
      <td>CLASS-784</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>74.262047</td>
      <td>CALIFORNIA REAL ESTATE INVESTMENT TRUST - Mar-...</td>
      <td>CLASS-784</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Making test data ready
#Let us create two new columns item_description1 and item_description2 from information from item_description
df_test['item_description1']=df_test['Item_Description'][1].split("-")[0]
df_test['item_description2']=df_test['Item_Description'][1].split("-")[3]
```


```python
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Inv_Id</th>
      <th>Vendor_Code</th>
      <th>GL_Code</th>
      <th>Inv_Amt</th>
      <th>Item_Description</th>
      <th>item_description1</th>
      <th>item_description2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>VENDOR-1197</td>
      <td>GL-6050100</td>
      <td>10.916343</td>
      <td>DESOTO INC - Jul-2008 - Store Management Real ...</td>
      <td>CENTURY REALTY TRUST</td>
      <td>Store Management Real Estate Real Estate Serv...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12</td>
      <td>VENDOR-792</td>
      <td>GL-6050100</td>
      <td>38.658772</td>
      <td>CENTURY REALTY TRUST - Nov-2019 - Store Manage...</td>
      <td>CENTURY REALTY TRUST</td>
      <td>Store Management Real Estate Real Estate Serv...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>VENDOR-792</td>
      <td>GL-6050100</td>
      <td>46.780476</td>
      <td>CENTURY REALTY TRUST - Jan-2006 - Store Manage...</td>
      <td>CENTURY REALTY TRUST</td>
      <td>Store Management Real Estate Real Estate Serv...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18</td>
      <td>VENDOR-792</td>
      <td>GL-6050100</td>
      <td>7.058866</td>
      <td>CENTURY REALTY TRUST - Sep-2002 - Store Manage...</td>
      <td>CENTURY REALTY TRUST</td>
      <td>Store Management Real Estate Real Estate Serv...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>VENDOR-792</td>
      <td>GL-6050100</td>
      <td>32.931765</td>
      <td>CENTURY REALTY TRUST - Nov-2018 - Store Manage...</td>
      <td>CENTURY REALTY TRUST</td>
      <td>Store Management Real Estate Real Estate Serv...</td>
    </tr>
  </tbody>
</table>
</div>




```python
gl_code_unique_test=df_test['GL_Code'].unique()
vendor_unique_test=df_test['Vendor_Code'].unique()
```


```python
#set each column
for i in range(0,len(df_test)):
    df_test['item_description1'][i] = df_test['Item_Description'][i].split("-")[0]
```

    C:\Users\Mayank\Anaconda3\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    


```python
for i in range(0,len(df_test)):
    df_test['item_description2'][i] = df_test['Item_Description'][i].split("-")[3]
```

    C:\Users\Mayank\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    


```python
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Inv_Id</th>
      <th>Vendor_Code</th>
      <th>GL_Code</th>
      <th>Inv_Amt</th>
      <th>Item_Description</th>
      <th>Product_Category</th>
      <th>item_description1</th>
      <th>item_description2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6.973473</td>
      <td>AETNA VARIABLE FUND - Apr-2002 - Store Managem...</td>
      <td>CLASS-784</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>25.053841</td>
      <td>AETNA VARIABLE FUND - Nov-2000 - Store Managem...</td>
      <td>CLASS-784</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>53.573737</td>
      <td>FAIRCHILD CORP - Nov-2001 - Store Management R...</td>
      <td>CLASS-784</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>67.388827</td>
      <td>CALIFORNIA REAL ESTATE INVESTMENT TRUST - Aug-...</td>
      <td>CLASS-784</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>74.262047</td>
      <td>CALIFORNIA REAL ESTATE INVESTMENT TRUST - Mar-...</td>
      <td>CLASS-784</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test['GL_Code']=df_test['GL_Code'].replace(gl_code_unique_test[0],0)
df_test['GL_Code']=df_test['GL_Code'].replace(gl_code_unique_test[1],1)
df_test['GL_Code']=df_test['GL_Code'].replace(gl_code_unique_test[2],2)
df_test['GL_Code']=df_test['GL_Code'].replace(gl_code_unique_test[3],3)
df_test['GL_Code']=df_test['GL_Code'].replace(gl_code_unique_test[4],4)
df_test['GL_Code']=df_test['GL_Code'].replace(gl_code_unique_test[5],5)
df_test['GL_Code']=df_test['GL_Code'].replace(gl_code_unique_test[6],6)
df_test['GL_Code']=df_test['GL_Code'].replace(gl_code_unique_test[7],7)
df_test['GL_Code']=df_test['GL_Code'].replace(gl_code_unique_test[8],8)
```


```python
des1_unique_test=df_test['item_description1'].unique()
des2_unique_test=df_test['item_description2'].unique()
```


```python
for i in range(len(vendor_unique_test)):
    df_test['Vendor_Code']=df_test['Vendor_Code'].replace(vendor_unique_test[i],i)
for j in range(len(des2_unique_test)):
    df_test['item_description2']=df_test['item_description2'].replace(des2_unique_test[j],j)
for k in range(len(des1_unique_test)):
    df_test['item_description1']=df_test['item_description1'].replace(des1_unique_test[k],k)
```


```python
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Inv_Id</th>
      <th>Vendor_Code</th>
      <th>GL_Code</th>
      <th>Inv_Amt</th>
      <th>Item_Description</th>
      <th>item_description1</th>
      <th>item_description2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>10.916343</td>
      <td>DESOTO INC - Jul-2008 - Store Management Real ...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>38.658772</td>
      <td>CENTURY REALTY TRUST - Nov-2019 - Store Manage...</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>1</td>
      <td>0</td>
      <td>46.780476</td>
      <td>CENTURY REALTY TRUST - Jan-2006 - Store Manage...</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>7.058866</td>
      <td>CENTURY REALTY TRUST - Sep-2002 - Store Manage...</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>1</td>
      <td>0</td>
      <td>32.931765</td>
      <td>CENTURY REALTY TRUST - Nov-2018 - Store Manage...</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X=df_train.drop(["Inv_Amt","Inv_Id","Item_Description"],axis=1)
```


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Vendor_Code</th>
      <th>GL_Code</th>
      <th>Product_Category</th>
      <th>item_description1</th>
      <th>item_description2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>CLASS-784</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>CLASS-784</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>CLASS-784</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>CLASS-784</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>CLASS-784</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train = X.drop("Product_Category", axis=1)
Y_train = X["Product_Category"]
X_test  = df_test.drop(["Inv_Amt","Inv_Id","Item_Description"], axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
```




    ((5719, 4), (5719,), (2292, 4))




```python
# support vector machine
svc=SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
```

    C:\Users\Mayank\Anaconda3\lib\site-packages\sklearn\svm\base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    




    99.79




```python
Y_pred
```




    array(['CLASS-784', 'CLASS-784', 'CLASS-784', ..., 'CLASS-75', 'CLASS-75',
           'CLASS-75'], dtype=object)




```python
submission = pd.DataFrame({
        "Inv_Id": df_test["Inv_Id"],
        "Product_Category": Y_pred
    })
submission.to_csv('submission_hack.csv', encoding='utf-8', index=False)
```


```python

```
