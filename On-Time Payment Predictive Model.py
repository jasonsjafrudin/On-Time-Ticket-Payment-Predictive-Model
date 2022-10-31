import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def blight_model():
    
    from sklearn.utils import resample
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler,MinMaxScaler 
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import auc
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_curve, auc
    
    # Loading the datasets
    df_train = pd.read_csv('./train.csv',encoding = "ISO-8859-1")
    df_test = pd.read_csv('./test.csv',encoding = "ISO-8859-1")
    df_latlong = pd.read_csv('./latlons.csv',encoding = "ISO-8859-1")
    df_add = pd.read_csv('./addresses.csv',encoding = "ISO-8859-1")
    

    # Removing Empty Features
    to_delete = ['violation_zip_code','non_us_str_code','grafitti_status']
    df_train = df_train.drop(to_delete,axis = 1)
    df_test = df_test.drop(to_delete,axis = 1)
    
    # Dealing with Feature Leakage
    to_delete = ['compliance_detail','payment_amount','payment_date','payment_status',
                'balance_due','collection_status','compliance_detail']
    df_train = df_train.drop(to_delete,axis = 1)
    
    # Dropping 'compliance' NULL Values
    df_train = df_train[df_train['compliance'].notnull()]
    
    # Re-balancing Samples
    df_majority = df_train[df_train['compliance']==0]
    df_minority = df_train[df_train['compliance']==1]
    
    # Unsampling Minority Class
    df_minority_upsampled = resample(df_minority, 
                                 replace=True,    
                                 n_samples=df_majority.shape[0],   
                                 random_state=123)
    
    # Combining Downsampled Majority Class with Minority Class
    df_train = pd.concat([df_majority, df_minority_upsampled])
    
    # Setting Index on 'ticket_id'
    df_train = df_train.set_index('ticket_id')
    df_test = df_test.set_index('ticket_id')
    
    # Replacing with Abbreviated agency names
    agencies = {'Buildings, Safety Engineering & Env Department':'BSEED',
               'Department of Public Works':'DPW',
               'Health Department':'HD',
               'Detroit Police Department':'DPD',
               'Neighborhood City Halls':'NCH'
               }

    df_train['agency_name'] = df_train['agency_name'].replace(agencies)
    df_test['agency_name'] = df_test['agency_name'].replace(agencies)
    
    # Formatting 'inspector_name'
    df_train['inspector_name'] = df_train['inspector_name'].str.lower()
    df_test['inspector_name'] = df_test['inspector_name'].str.lower()
    
    # bins by Count of Tickets
    df_train['active_inspector'] = df_train['inspector_name'].replace(df_train['inspector_name'].value_counts().to_dict())
    df_test['active_inspector'] = df_test['inspector_name'].replace(df_test['inspector_name'].value_counts().to_dict())
    
    df_train['active_inspector_bins'], bins_train = pd.qcut(df_train['active_inspector'],q=6,
                                                       labels=['Inactive','Very_Low','Low','Average','High','Very_High'],retbins=True)
    bins_train[0] = 0.000
    
    df_test['active_inspector_bins'] = pd.cut(df_test['active_inspector'],bins=bins_train,
                                          labels=['Inactive','Very_Low','Low','Average','High','Very_High'],right=True)
    
    # Dropping 'inspector_name'
    df_train = df_train.drop(['inspector_name'],axis=1)
    df_test = df_test.drop(['inspector_name'],axis=1)
    
    # Dropping 'violator_name'
    df_train = df_train.drop(['violator_name'],axis=1)
    df_test = df_test.drop(['violator_name'],axis=1)
    
    # Creating New Feature: 'is_home'
    df_train['is_home'] = df_train['violation_street_name']==df_train['mailing_address_str_name']
    df_test['is_home'] = df_test['violation_street_name']==df_test['mailing_address_str_name'] 
    
    # Creating New Feature: 'has_address'
    df_train['has_address'] = ~df_train['mailing_address_str_number'].isnull()
    df_test['has_address'] = ~df_test['mailing_address_str_number'].isnull()
    
    # Dropping 'mailing_address_str_number'
    df_train = df_train.drop(['mailing_address_str_number'],axis=1)
    df_test = df_test.drop(['mailing_address_str_number'],axis=1)
    
    # Dropping 'mailing_address_str_name'
    df_train = df_train.drop(['mailing_address_str_name'],axis=1)
    df_test = df_test.drop(['mailing_address_str_name'],axis=1)
    
    # Creating New Features:
    # 'is_local' (Michigan)
    # 'is_neighbor' (Wisconsin, Illinois, Indiana, Ohio)
    # 'is_foreign' (Non U.S.)
    
    df_train['is_local'] = df_train['state'] == 'MI'
    df_test['is_local'] = df_test['state'] == 'MI'
    
    neighbor_state = ['WI','IL','IN','OH']
    df_train['is_neigh'] = df_train['state'].isin(neighbor_state)
    df_test['is_neigh'] = df_test['state'].isin(neighbor_state)
    
    df_train['is_foreign'] = df_train['country']!='USA'
    df_test['is_foreign'] = df_test['country']!='USA'
    
    # Setting Missing State for foreign countries (Non U.S.) 
    df_train.loc[df_train['state'].isnull(),'state']='XX'
    df_test.loc[df_test['state'].isnull(),'state']='XX'
    
    # Formatting 'city'
    df_train['city'] = df_train['city'].str.lower()
    df_test['city'] = df_test['city'].str.lower()
    
    # Find NULL 'city' for df_test
    df_test.loc[df_test['city'].isnull(),'city']='westerville'
    
    # Dates Transformations
    df_train['ticket_issued_date'] = pd.to_datetime(df_train['ticket_issued_date'])
    df_test['ticket_issued_date'] = pd.to_datetime(df_test['ticket_issued_date'])
    
    df_train['ticket_issued_month'] = df_train['ticket_issued_date'].dt.month
    df_test['ticket_issued_month'] = df_test['ticket_issued_date'].dt.month
    
    df_train['ticket_issued_start'] = df_train['ticket_issued_date'].dt.day<=15
    df_test['ticket_issued_start'] = df_test['ticket_issued_date'].dt.day<=15
    
    # Creating New Feature: 'disposition_bins'
    disposition_bins = [
        'Responsible by Default',
        'Responsible by Admission',
        'Responsible by Determination',
        'Responsible (Fine Waived) by Deter']
    
    df_train['disposition_bins'] = df_train['disposition'].apply(lambda x: x if x in disposition_bins else 'Other')
    df_test['disposition_bins'] = df_test['disposition'].apply(lambda x: x if x in disposition_bins else 'Other')
    
    # Creating New Feature: 'has_hearing'
    df_train['has_hearing'] = ~df_train['hearing_date'].isnull()
    df_test['has_hearing'] = ~df_test['hearing_date'].isnull()
    
    df_train['violation_code'] = (df_train['violation_code'].str.replace('(','')).str.replace(')','').str.replace(' ','').str.replace('-','').str.extract('(\d+)',expand=True).apply(lambda x: x[0:8])
    df_test['violation_code'] = (df_train['violation_code'].str.replace('(','')).str.replace(')','').str.replace(' ','').str.replace('-','').str.extract('(\d+)',expand=True).apply(lambda x: x[0:8])
    
    to_drop = ['violation_street_number','zip_code',
               'hearing_date','ticket_issued_date',
               'violation_description','active_inspector',
              'city','country','violation_street_name',
              'disposition']
    df_train = df_train.drop(to_drop,axis=1)
    df_test = df_test.drop(to_drop,axis=1)
    
    # Setting Index on 'ticket_id'
    df_add = df_add.set_index('ticket_id')

    # Merge
    df_train = pd.merge(left=df_train,right=df_add,how='left',left_index=True,right_index=True)
    df_test = pd.merge(left=df_test,right=df_add,how='left',left_index=True,right_index=True)
    
    # Setting Index on 'address'
    df_latlong = df_latlong.set_index('address')

    # Merge
    df_train = pd.merge(left=df_train,right=df_latlong,how='left',left_on='address',right_index=True)
    df_test = pd.merge(left=df_test,right=df_latlong,how='left',left_on='address',right_index=True)

    # Dropping 'address'
    df_train = df_train.drop(['address'],axis=1)
    df_test = df_test.drop(['address'],axis=1)
    
    step = 0.03
    to_bin = lambda x: str(np.floor(x / step) * step)
    df_train["latbin"] = df_train['lat'].map(to_bin)
    df_train["lonbin"] = df_train['lon'].map(to_bin)

    df_test["latbin"] = df_test['lat'].map(to_bin)
    df_test["lonbin"] = df_test['lon'].map(to_bin)
    
    df_train_dum = pd.get_dummies(data=df_train,
                              columns=['agency_name','state','disposition_bins',
                                      'active_inspector_bins','ticket_issued_month',
                                       'ticket_issued_start','violation_code','latbin','lonbin'],
                              drop_first=False)

    df_test_dum = pd.get_dummies(data=df_test,
                              columns=['agency_name','state','disposition_bins',
                                      'active_inspector_bins','ticket_issued_month',
                                       'ticket_issued_start','violation_code','latbin','lonbin'],
                              drop_first=False)

    # Dropping Column Differences
    df_test = df_test_dum.drop(['disposition_bins_Other','lonbin_-82.89','lat','lon'],axis=1)

    df_train = df_train_dum.drop(['agency_name_HD', 'agency_name_NCH', 'compliance',
                                  'latbin_42.12', 'latbin_42.15', 'latbin_42.51',
                                  'latbin_42.57', 'latbin_42.6', 'latbin_42.63',
                                  'latbin_42.66', 'latbin_42.69', 'latbin_42.72',
                                  'latbin_42.75', 'latbin_42.9', 'latbin_44.76',
                                  'lonbin_-82.53', 'lonbin_-83.34', 'lonbin_-83.46',
                                  'lonbin_-83.49', 'lonbin_-83.55', 'lonbin_-83.76',
                                  'lonbin_-84.12', 'lonbin_-84.42', 'state_BL',
                                  'violation_code_6163', 'violation_code_9136',
                                  'compliance','lat','lon'],axis=1)

    Y_train = df_train_dum['compliance']

    X_train = df_train.copy()
    X_test = df_test.copy()

    stdsc = StandardScaler().fit(X_train)
    X_train = stdsc.transform(X_train)
    X_test = stdsc.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(X_train,Y_train)


    rdf = RandomForestClassifier(n_estimators=200,oob_score = True)
   
    rdf.fit(X_train,y_train)

    y_score_rdf = rdf.predict_proba(X_val)
    
    y_test_proba = rdf.predict_proba(X_test)[:,1]
    
    result = pd.Series(y_test_proba,index=df_test.index)
    
    return result


blight_model()






