import scipy.io as sio
import scipy.sparse as spp
import numpy as np
import scipy as sp
from sklearn.preprocessing import normalize
import json
from collections import defaultdict
import os
import pandas as pd
import ast
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding



business_file = './yelp/business.json'
review_file = './yelp/review.json'
user_file = './yelp/user.json'
tip_file = './yelp/tip.json'




def get_buss_feature(filename):
    business_data = []
    with open(filename) as f:
        for line in f:
            business_data.append(json.loads(line))
    business_df = pd.DataFrame.from_dict(business_data)
    # "name" column name is ambiguous with df.name - change it
    business_df = business_df.rename(columns = {'name': 'BusinessName'})
    business_df['categories_clean'] = list(map(lambda x: '|'.join(x), business_df['categories']))
    categories_df = business_df.categories_clean.str.get_dummies(sep='|')
    # merge
    business_df = business_df.merge(categories_df, left_index=True, right_index=True)
    # remove intermediate columns (no longer needed)
    business_df.drop(['categories', 'categories_clean'], axis=1, inplace=True)
    business_df = business_df.join(pd.DataFrame(business_df['attributes'].to_dict()).T)
    # further split sub-attributes into their own columns
    cols_to_split = ['BusinessParking', 'Ambience', 'BestNights', 'GoodForMeal', 'HairSpecializesIn', 'Music']
    for col_to_split in cols_to_split:
        new_df = pd.DataFrame(business_df[col_to_split].to_dict()).T
        new_df.columns = [col_to_split + '_' + str(col) for col in new_df.columns]
        business_df = business_df.join(new_df)

    business_df.drop(['attributes'] + cols_to_split, axis=1, inplace=True)
    # columns with non-boolean categorical values:
    cols_to_split = ['AgesAllowed', 'Alcohol', 'BYOBCorkage', 'NoiseLevel', 'RestaurantsAttire', 'Smoking', 'WiFi']
    new_cat = pd.concat([pd.get_dummies(business_df[col], prefix=col, prefix_sep='_') for col in cols_to_split], axis=1)
    # keep all columns (not n-1) because 0's for all of them indicates that the data was missing (useful info)
    business_df = pd.concat([business_df, new_cat], axis=1)
    business_df.drop(cols_to_split, inplace=True, axis=1)
    business_df = business_df.fillna(0.5).apply(pd.to_numeric, errors='ignore')  # can narrow with .iloc[:,648:722] if necessary
    business_df['postal_code'] = business_df['postal_code'].fillna(0)

    # check that all nulls are removed
    business_df.isnull().sum().sum()
    def com(x):
        if x<=10:
            return 1
        else:
            return 0

    def com1(x):
        if x>10:
            if x<=100:
                return 1
        return 0 

    def com2(x):
        if x>1000:
            return 1
        return 0  


    df_re =  pd.DataFrame()
    df_re['review_10'] = list(map(com, business_df['review_count'] ))
    df_re['review_100'] = list(map(com1, business_df['review_count'] ))
    df_re['review_1000'] = list(map(com2, business_df['review_count'] ))
    attribute_cols = business_df.columns[648:737]
    df_attri = business_df[attribute_cols].to_numpy()
    embedding = LocallyLinearEmbedding(n_components=4)
    rest_feature = embedding.fit_transform(df_attri)
    re = df_re.to_numpy()
    new_r = np.concatenate((re, rest_feature), axis=1)
    print(new_r.shape)
   
    np.save("./rest_feature.npy", new_r)
    re_index = pd.DataFrame(business_df['business_id'])
    re_index['id'] = range(len(business_df)) 
    re_index = re_index.to_numpy()
    np.save('./rest_index.npy', re_index)

    

def get_user_feature(filename):
    business_data = []
    with open(filename) as f:
        for line in f:
            business_data.append(json.loads(line))
    business_df = pd.DataFrame.from_dict(business_data)
    new_df = business_df.drop(['friends', 'user_id', 'yelping_since', 'name', 'elite'], axis=1)
    new_df = new_df.to_numpy()
    embedding = LocallyLinearEmbedding(n_components=10)
    X_transformed = embedding.fit_transform(new_df)
    
    user_index = pd.DataFrame(business_df['user_id'])
    user_index['id'] = range(len(business_df)) 
    user_index = user_index.to_numpy()
    np.save("./embedding_user.npy", X_transformed)
    np.save('./user_index.npy', user_index)


# BUSINESS.JSON
business_data = []
#print 'Reading business.json'
with open(business_file,encoding="utf8") as f:
    for line in f:
        business_data.append(json.loads(line))
business_df = pd.DataFrame.from_dict(business_data)
business_sample = business_df[business_df['city'].str.contains('leveland')]
#print 'Creating business_sample.json'
with open('business_sample.json', 'w') as f:
    business_sample.to_json(f, orient='records', lines=True)
# now that sample file is made, get list of business id's from it
# convert into a dict with values as keys for fast searching
#print 'Getting business_ids'
business_ids = pd.Series(business_sample['business_id'].index.values, index=business_sample['business_id']).to_dict()

# CHECKIN.JSON
checkin_file = './yelp/checkin.json'

checkin_data = []
with open(checkin_file, encoding="utf8") as f:
    for line in f:
        newline = ast.literal_eval(line)
        if newline['business_id'] in business_ids:
            checkin_data.append(json.loads(line))
checkin_df = pd.DataFrame.from_dict(checkin_data)
with open('checkin_sample.json', 'w') as f:
    checkin_df.to_json(f, orient='records', lines=True)

    
    
filename = "./yelp/business_sample_cleveland.json"
filename1 = "./yelp/user_sample_cleveland.json"

get_buss_feature(filename)
get_user_feature(filename1)