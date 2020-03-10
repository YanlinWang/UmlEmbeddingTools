# !-*-coding:utf-8 -*-
# @Author : Lun

import os
import sys
import json
import argparse
import numpy as np
import pickle
import networkx as nx
import re
import decamelize
from collections import namedtuple
import pdb

import tensorflow as tf
import tensorflow_hub as hub

import torch
from torch_geometric.data import Data


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
#ROOT_PATH = os.path.join(FILE_PATH, "..")
ROOT_PATH = FILE_PATH
DATA_PATH = os.path.join(ROOT_PATH, "data")
RES_PATH = os.path.join(ROOT_PATH, "res")
if os.path.exists(RES_PATH) == False:
    os.mkdir(RES_PATH)
    
URLs = {
    "dan": "https://tfhub.dev/google/universal-sentence-encoder/4",
    "transformer": "https://tfhub.dev/google/universal-sentence-encoder-large/5"
}

NUM_TYPE = 4
REL_MAP = {
    "DEPEND": 0,
    "NAVASSOC": 1,
    "ASSOC": 1,
    "IMPLEMENTS": 2,
    "EXTENDS": 3,
    "COMPOSED": 1,
    "NAVCOMPOSED": 1
}

def preprocess(s):
    ProcessRes = namedtuple("ProcessRes",['string','is_template','is_sub'])
    is_template = (re.search(r"<.*>", s) is not None)
    is_sub = (re.search(r"\.", s) is not None)
    a = decamelize.convert(s).replace('_', ' ')
    a = re.sub(r"<.*>|\.", "", a)
    return ProcessRes(string = a, is_template = is_template, is_sub = is_sub)

def embedding(args, data):
    
    print("[...] Start loading embedding model.")
    if args.gpu:
        embed = hub.load(URLs[args.embedding_method])
    else:
        with tf.device(r'/cpu:0'):
            embed = hub.load(URLs[args.embedding_method])
    print("[X] Loaded embedding model.")
        
        
    num_output = 30
    for idx, i in enumerate(data):
        if (idx % num_output == 0):
            print("[...] Handling %d to %d class names" % (idx, idx + num_output))
        classes = [item["class_declaration"]["name"] for item in data[i]['nodes_information'].values()]
        mapp = [key for key in data[i]["nodes_information"]]
        res = [preprocess(string) for string in classes]
        strings = [item.string for item in res]
        
        if args.gpu:
            vectors = embed(strings).numpy()
        else:
            with tf.device(r'/cpu:0'):
                vectors = embed(strings).numpy()
            
        data[i]["processed_info"] = {"embedding" : vectors, 
                                     "is_template" : [item.is_template for item in res], 
                                     "is_sub" : [item.is_sub for item in res],
                                     "mapping" : mapp}
    
    print("[X] Preocessed the class names")
    # print(data)
    print("[...] Saving")
    if args.output is None:
        outpath = os.path.join(RES_PATH, "uml_embedding.pkl")
    else:
        outpath = os.path.join(args.output, "uml_embedding.pkl")
    with open(outpath, "wb") as f:
        pickle.dump(data, f)
    print("[X] Saved.")
    return data
        

def transfer(args, data):
    dataset = {}
    num_output = 10000
    for idx, i in enumerate(data):
        if (idx % num_output == 0):
            print("[...] Handling %d to %d graphs" % (idx, idx + num_output))
        mapp = data[i]["processed_info"]["mapping"]
        rmapp = {item: idx for idx, item in enumerate(mapp)}
        n = data[i]["number_of_nodes"]
        
        is_template = np.array([1.0 if item else 0.0 for item in data[i]["processed_info"]["is_template"]]).reshape(n, 1)
        x = np.concatenate([data[i]["processed_info"]["embedding"], is_template], axis = 1)
        x = torch.from_numpy(x).to(torch.float32)
        try:
            edge_index = torch.tensor([[rmapp[edge[k]] for edge in data[i]['edge_info']] for k in range(2)], dtype=torch.long)
        except KeyError:
            pdb.set_trace()
        
        #for edge in data[i]["edge_info"]:
        #    if edge[2]["relationtype"] == "COMPOSED" or edge[2]["relationtype"] == "NAVCOMPOSED":
        #        print(i, edge[2]["relationtype"])
        edge_type = torch.tensor([REL_MAP[edge[2]["relationtype"]] for edge in data[i]["edge_info"]], dtype = torch.long)
        ones = torch.sparse.torch.eye(NUM_TYPE)
        edge_type = ones.index_select(0, edge_type)
        
        
        data_pyg = Data(x=x, edge_index = edge_index, edge_attr = edge_type)
        dataset[i] = data_pyg
        
    print("[...] Saving")
    if args.output is None:
        outpath = os.path.join(RES_PATH, "uml_dataset.pt")
    else:
        outpath = os.path.join(args.output, "uml_dataset.pt")
    torch.save(dataset, outpath)
    print("[X] Saved.")
        

def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--data', type = str, required = True)
    parser.add_argument('--output', type = str)
    parser.add_argument('--operation', type = str, default = "all", help = "[embedding | transfer | all]")
    parser.add_argument('--gpu', type = bool, default = True)
    parser.add_argument('--embedding_method', type = str, default = "transformer", help = r"[transformer | dan]")
    args = parser.parse_args()
    
    data_path = os.path.join(DATA_PATH, args.data)
    with open(data_path, "rb") as f:
        data = pickle.load(f)
        
    if (args.operation == 'all'):
        data = embedding(args, data)
        transfer(args, data)
    elif (args.operation == 'embedding'):
        data = embedding(args, data)
    elif (args.operation == 'transfer'):
        transfer(args, data)
    else:
        print("Operation isn't supported.")
    
    
    
    

if __name__ == "__main__":
    main()
