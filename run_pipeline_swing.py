
import os
from wave import Wave_write
import pandas as pd
import numpy as np
import pickle

import argparse
from datetime import datetime
import math
from collections import defaultdict
from scipy.sparse import coo_matrix, csr_matrix
from scipy import sparse

from util_vCF.helper import (
    load_target_items,
    load_test_data,
    load_train_data_with_coo
)



def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--item_id_START", required=True)
    parser.add_argument("--item_id_END", required=True)
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--output_csv_path", required=True)
    parser.add_argument("--output_pickle_path", required=True)
    return parser

    
    
def load_edge_df(data_path):
    """Loads the Graph Edge data from the Input Source
    Args:
        edge_list_data_path (str): path to the csv/tsv file containing the training data
        subsample_num_user (int, optional): number of users to subsample. Defaults to None.

    Returns:
        DataFrame: Edge List DataFrame
    """
    df = pd.read_table(data_path, sep=',')
    
    if len(df.columns) == 3:
        df.columns = ['uid', 'tid', 'ln_coclick_count']
    elif len(df.columns) == 2:
        df.columns = ['uid', 'tid']
    return df
        

def main():
    parser = get_arg_parser()
    args, _ = parser.parse_known_args()
    d = vars(args)
    TRAIN_CSV = d['train_csv']
    ITEM_ID_START = int(d['item_id_START'])
    ITEM_ID_END = int(d['item_id_END'])
    ALPHA = float(d['alpha'])
    OUTPUT_CSV_PATH = d['output_csv_path']
    OUTPUT_PICKLE_PATH = d['output_pickle_path']

    
    args= {}
    args['input_path'] =  TRAIN_CSV
    args['input_delimiter'] = ','
    args['rating_scale'] = 1.0
    args['rating_offset'] = 0.0
    args['weight_choice'] = 'ctr'
    
    print("Loading Training Data")
    train_user_item_coo, train_user_item_csr, train_df, item_number, idfs, item_sums, _ = load_train_data_with_coo(
        args['input_path'], args['input_delimiter'], args['rating_scale'], args['rating_offset'], args['weight_choice']
    )
    
    num_users = train_user_item_coo.shape[0]
    num_items = train_user_item_coo.shape[1]
    
    print("Number of Users: {}".format(num_users))
    print("Number of Items: {}".format(num_items))
    print("Number of Interactions: {}".format(train_user_item_coo.nnz))

    swing_dict, swing_edge_list_df = get_swing_i2imatrix_csr(train_user_item_csr, ITEM_ID_START, ITEM_ID_END, ALPHA)
    print("Swing Algorithm Complete.")

    print("Saving swing dict as pickle.....")
    # Save the swing dict as pickle
    with open(OUTPUT_PICKLE_PATH, 'wb') as f:
        pickle.dump(swing_dict, f)
    print("Saving swing dict as pickle complete.")
    
    #Save the swing edge list as csv
    print("Saving swing edge list as csv.....")
    swing_edge_list_df.to_csv(OUTPUT_CSV_PATH, index=False, header=False)
    print("Saving swing edge list as csv complete.")
    
    num_swing_score_edges = len(swing_edge_list_df)
    print("Number of Swing Score Edges/ length of generated csv: {}".format(num_swing_score_edges))
    

    #print("Swing Matrix Shape: {}".format(swing_i2imatrix_csr.shape))
    
    
    
def get_swing_i2imatrix_csr(train_user_item_csr, ITEM_ID_START = 0, ITEM_ID_END = -1, alpha=0.001):
    
    num_users, num_items = train_user_item_csr.shape
    
    if ITEM_ID_END == -1:
        ITEM_ID_END = num_items - 1

    
    weight = np.zeros(num_users)
    items = defaultdict(set)
    users = defaultdict(list)

    train_user_item_coo = train_user_item_csr.tocoo()
    num_data = len(train_user_item_coo.data)
    for n in range(num_data):
        items[train_user_item_coo.row[n]].add(train_user_item_coo.col[n])
        users[train_user_item_coo.col[n]].append(train_user_item_coo.row[n])

    for u in range(num_users):
        weight[u] = 1 / math.sqrt(len(items[u]))

    swing_dict = defaultdict(int)
    
    for i in range(ITEM_ID_START, ITEM_ID_END + 1):
        
        if i%10000 == 0:
            print(f"{i} items processed")
            
        for p, u in enumerate(users[i]):
            for v in users[i][p + 1 :]:
                common_items_uv = items[u] & items[v]
                common_items_uv.remove(i)
                k = len(common_items_uv)
                for j in common_items_uv:
                    swing = weight[u] * weight[v] * 1 / (alpha + k)
                    swing_dict[i, j] += swing
                    #swing_dict[j, i] += swing
                    
    keys = list(swing_dict.keys())
    rows, cols = list(zip(*keys))
    data = list(swing_dict.values())

    swing_edge_list_df = pd.DataFrame()
    swing_edge_list_df['uid'] = rows
    swing_edge_list_df['tid'] = cols
    swing_edge_list_df['weight'] = data

    return swing_dict, swing_edge_list_df



if __name__ == "__main__":
    main()