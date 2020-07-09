import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sn
import pandas as pd
from collections import defaultdict

layers = ["0", "3", "6", "6-random0", "6-random1", "6-random2", "6-random3", "6-random4", "9", "12"]
#layers = ["0", "3"]

def load_data():

 layer2data = defaultdict(dict)
 layer2projs = defaultdict(dict)
 
 for layer in layers:

    for masked in ["True"]:
    
        fname = "../data/datasets.5000a.layer={}.masked={}.pickle".format(layer, masked)
        fname2 = "../data/datasets.5000t.layer={}.masked={}.pickle".format(layer, masked)

        with open(fname, "rb") as f:
            train_dev_lex = pickle.load(f)
    
        with open(fname2, "rb") as f:
            train_dev_nonlex = pickle.load(f)

        projs_path = "../data/type2P.layer={}.iters=16.classifier=sgd.masked={}.pickle".format(layer, masked)
        with open(projs_path, "rb") as f:
            type2proj = pickle.load(f)
    
        layer2projs[layer] = type2proj
        
        comps = projs_path.split(".")
        iters = comps[-4]
        
        
        dev_x_lex, dev_y_lex = np.concatenate([train_dev_lex[positive_type]["dev"][0] for positive_type in train_dev_lex.keys()], axis = 0), np.concatenate([train_dev_lex[positive_type]["dev"][1] for positive_type in train_dev_lex.keys()], axis = 0)
        dev_type_lex = []
        for rc_type in train_dev_lex.keys():
        
            dev_type_lex.extend([rc_type]*len(train_dev_lex[rc_type]["dev"][0]))
        dev_type_lex = np.array(dev_type_lex)
        
        
        dev_x_nonlex, dev_y_nonlex = np.concatenate([train_dev_nonlex[positive_type]["dev"][0] for positive_type in train_dev_nonlex.keys()], axis = 0), np.concatenate([train_dev_nonlex[positive_type]["dev"][1] for positive_type in train_dev_nonlex.keys()], axis = 0)
        dev_type_nonlex = []
        for rc_type in train_dev_nonlex.keys():
        
            dev_type_nonlex.extend([rc_type]*len(train_dev_nonlex[rc_type]["dev"][0]))
        
        dev_type_nonlex = np.array(dev_type_nonlex)
        
        
        relevant_lex_mask = dev_y_lex == 1
        relevant_nonlex_mask  = dev_y_nonlex == 1
        dev_x_nonlex, dev_y_nonlex, dev_type_nonlex = dev_x_nonlex[relevant_nonlex_mask], dev_y_nonlex[relevant_nonlex_mask], dev_type_nonlex[relevant_nonlex_mask]
        dev_x_lex, dev_y_lex, dev_type_lex = dev_x_lex[relevant_lex_mask], dev_y_lex[relevant_lex_mask], dev_type_lex[relevant_lex_mask]
        
        
        assert len(dev_x_nonlex) == len(dev_y_nonlex) == len(dev_type_nonlex)
        assert len(dev_x_lex) == len(dev_y_lex) == len(dev_type_lex)
        
        layer2data[layer]["lex"] = (dev_x_lex, dev_y_lex, dev_type_lex)
        layer2data[layer]["nonlex"] = (dev_x_nonlex, dev_y_nonlex, dev_type_nonlex)
        
        print(dev_x_nonlex.shape[0], dev_x_lex.shape[0])

        
 return layer2data, layer2projs
 
 

def collect_vecs():

    layer2data, layer2projs = load_data()
    
    vecs_lex, vecs_rowspace, labels = [],[], []
    vecs_rowspace_nonlex = []

    
    layer2ttype2vecs_rowspace = defaultdict(dict)
    layer2type2vecs_rowspace_nonlex = defaultdict(dict)
    layer2ttype2vecs_nullspaces = defaultdict(dict)
    layer2type2vecs_nullspace_nonlex = defaultdict(dict)
        
    for layer in layer2data.keys():
    
        type2vecs_rowspace = {}
        type2vecs_rowspace_nonlex = {}
        
        for rc_type in ["src", "orc", "orrc", "prc", "prrc", "all"]:
        
            P_rowspace = np.eye(768) - layer2projs[layer][rc_type]
            
            lex_x, lex_y, lex_type = layer2data[layer]["lex"]
            nonlex_x, nonlex_y, nonlex_type = layer2data[layer]["nonlex"]
            
            if rc_type == "all":            
                mask_lex = np.ones_like(lex_type).astype(bool)
                mask_nonlex = np.ones_like(nonlex_type).astype(bool)
                
            else:
                mask_lex = lex_type == rc_type
                mask_nonlex = nonlex_type == rc_type
           

            type2vecs_rowspace[rc_type] = lex_x[mask_lex].dot(P_rowspace)
            type2vecs_rowspace_nonlex[rc_type] = nonlex_x[mask_nonlex].dot(P_rowspace)
            
        layer2ttype2vecs_rowspace[layer] = type2vecs_rowspace
        layer2type2vecs_rowspace_nonlex[layer] = type2vecs_rowspace_nonlex
        
    return layer2ttype2vecs_rowspace, layer2type2vecs_rowspace_nonlex
    
      


def calc_sims(layer2ttype2vecs_rowspace, layer2type2vecs_rowspace_nonlex):

    
    type2ind = {d:i for i,d in enumerate(layer2type2vecs_rowspace_nonlex["0"].keys())}
    ind2type = {i:d for d,i in type2ind.items()}
    layer2sims = dict()
    
    for layer in layer2ttype2vecs_rowspace.keys():
    
        type2vecs_rowspace, type2vecs_rowspace_nonlex =  layer2ttype2vecs_rowspace[layer], layer2type2vecs_rowspace_nonlex[layer]
        sims = np.zeros((6,6))
    
        from sklearn.metrics.pairwise import cosine_similarity

        for key, vecs in type2vecs_rowspace.items():
            for key2, vecs2 in type2vecs_rowspace_nonlex.items():

        
                sims2 = cosine_similarity(vecs, vecs2)
                sims[type2ind[key], type2ind[key2]] = np.mean(sims2) #mean1_normed.dot(mean2_normed.T)
                
        layer2sims[layer] = sims

    labels = [ind2type[i] for i in range(len(ind2type))]
    for layer, sims in layer2sims.items():
        df = pd.DataFrame(sims, index = labels, columns = labels)
        print("Layer {}".format(layer))
        print(df)
        print("========================================================")
            
layer2ttype2vecs_rowspace, layer2type2vecs_rowspace_nonlex = collect_vecs()        
calc_sims(layer2ttype2vecs_rowspace, layer2type2vecs_rowspace_nonlex)   
    
    
    
