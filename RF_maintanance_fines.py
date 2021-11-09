# -*- coding: utf-8 -*-
"""
Random Forest to predict whether a given blight ticket will be paid on time
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

df_train = pd.read_csv('train.csv', encoding = "ISO-8859-1", low_memory=False)
df_test = pd.read_csv('test.csv', encoding = "ISO-8859-1", low_memory=False)
addresses = pd.read_csv('addresses.csv', encoding = "ISO-8859-1", low_memory=False)
latlons = pd.read_csv('latlons.csv', encoding = "ISO-8859-1", low_memory=False)

addresses=pd.merge(addresses,latlons, how='left',on='address')

addresses.set_index('address', inplace=True)
addresses.at['20424 bramford, Detroit MI', 'lon'] = -83.023182
addresses.at['20424 bramford, Detroit MI', 'lat'] = 42.44667
addresses.at['8325 joy rd, Detroit MI 482O4', 'lon'] = -83.151228
addresses.at['8325 joy rd, Detroit MI 482O4', 'lat'] = 42.358858
addresses.at['1201 elijah mccoy dr, Detroit MI 48208', 'lon'] = -83.080371
addresses.at['1201 elijah mccoy dr, Detroit MI 48208', 'lat'] = 42.35853
addresses.at['12038 prairie, Detroit MI 482O4', 'lon'] = -83.143074
addresses.at['12038 prairie, Detroit MI 482O4', 'lat'] = 42.376722
addresses.at['6200 16th st, Detroit MI 482O8', 'lon'] = -83.095686
addresses.at['6200 16th st, Detroit MI 482O8', 'lat'] = 42.359923
addresses.at['445 fordyce, Detroit MI', 'lon'] = -83.051460
addresses.at['445 fordyce, Detroit MI', 'lat'] = 42.328590
addresses.at['8300 fordyce, Detroit MI', 'lon'] = -83.058189
addresses.at['8300 fordyce, Detroit MI', 'lat'] = 42.383251
addresses.reset_index(inplace=True)

# for feature processing
remove_cols=['ticket_id','address','agency_name', 'inspector_name', 'violator_name', 'violation_street_number', 'violation_street_name', 'mailing_address_str_number',
    'mailing_address_str_name','city', 'state','country', 'zip_code', 'non_us_str_code','ticket_issued_date', 'hearing_date',
    'grafitti_status','violation_zip_code']

transform_cols=['violation_code', 'violation_description', 'disposition']

# Train set
#df_train = df_train[~df_train['compliance'].isnull()]
X_df=df_train[df_test.columns]
X_df=pd.merge(X_df,addresses, how='left',on='ticket_id')
#X_df=X_df.iloc[:,1:]
X_df.drop(remove_cols, axis=1, inplace=True)
X_df.at[X_df['fine_amount'].isnull(),'fine_amount']=0

Y_df=df_train.iloc[:,-1]==1

# Target set
df_test=pd.merge(df_test,addresses, how='left',on='ticket_id')
Target_id=df_test[['ticket_id']]
X_target=df_test.drop(remove_cols, axis=1)

# feature processing
le = preprocessing.LabelEncoder()
X_df_trans=X_df
X_target_trans=X_target

for i in transform_cols:
    le.fit(X_df_trans[i].append(X_target_trans[i]))
    X_df_trans[i] = le.transform(X_df_trans[i])
    X_target_trans[i] = le.transform(X_target_trans[i])


    #classifier training+APPLICATION 
clf = RandomForestClassifier(random_state=0).fit(X_df, Y_df)
Target_prop=clf.predict_proba(X_target)
out=pd.Series(Target_prop[:,1], index=Target_id.squeeze(), name='compliance')