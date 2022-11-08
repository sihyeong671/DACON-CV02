import argparse
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing

from Modules import *

def voting(inputs):
    artist = np.zeros(50)
    for input in inputs:
        input = list(map(float, input[1:-2].split(', ')))
        artist += np.array(input)
    
    return artist.argmax()
    
def soft_voting(args):
    file_names = os.listdir(args.save_path)
    df_main = pd.DataFrame()
    df_temp = pd.read_csv(os.path.join(args.save_path, file_names[0]))
    df_main['id'] = df_temp['id']
    for idx, file_name in enumerate(file_names):
        df_temp = pd.read_csv(os.path.join(args.save_path, file_name))
        df_main[f'ans{idx}'] = df_temp['artist']

    artist_list = []
    for i in range(len(df_main)):
        data = df_main.loc[i].values
        artist_list.append(voting(data[1:]))
    
    df = pd.read_csv(os.path.join('./data', 'train_repaired.csv'))
    le = preprocessing.LabelEncoder()
    df['artist'] = le.fit_transform(df['artist'].values)
    
    preds = le.inverse_transform(artist_list)
    submit = pd.read_csv(os.path.join('./data', 'sample_submission.csv'))

    submit['artist'] = preds

    submit.to_csv(os.path.join(args.save_path , args.save_name), index=False)    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='./data/vote')
    parser.add_argument('--save_name', default='final_submission.csv')
    args = parser.parse_args()
    
    soft_voting(args)

    
    