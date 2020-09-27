import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sn
import pandas as pd
from collections import defaultdict
import argparse
import sys
sys.path.append("inlp/")
from inlp import debias

pd.set_option('precision', 5)

layers = ["0", "3", "6", "6-random0", "6-random1", "6-random2", "6-random3", "6-random4", "9", "12"]
#layers = ["0", "3"]

def load_data(classifier, iters):

 layer2data = defaultdict(dict)
 layer2projs = defaultdict(dict)
 layer2ws = defaultdict(dict)
 
 for layer in layers:

    for masked in ["True"]:
    
        fname = "../data/datasets.adapt.layer={}.masked={}.model=bert.pickle".format(layer, masked)
        fname2 = "../data/datasets.test.layer={}.masked={}.model=bert.pickle".format(layer, masked)

        with open(fname, "rb") as f:
            train_dev_lex = pickle.load(f)
    
        with open(fname2, "rb") as f:
            train_dev_nonlex = pickle.load(f)

        projs_path = "../data/type2P.layer={}.iters={}.classifier={}.masked={}.pickle".format(layer, iters, classifier, masked)
        with open(projs_path, "rb") as f:
            type2proj = pickle.load(f)
    
        layer2projs[layer] = type2proj
        
        ws_path = "../data/type2W.layer={}.iters={}.classifier={}.masked={}.pickle".format(layer, iters, classifier, masked)
        with open(ws_path, "rb") as f:
            type2ws = pickle.load(f)
    
        layer2ws[layer] = type2ws
                
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

 return layer2data, layer2projs, layer2ws
 
 

def collect_vecs(classifier, iters, do_random_projection):

    layer2data, layer2projs, layer2ws = load_data(classifier, iters)
    
    vecs_lex, vecs_rowspace, labels = [],[], []
    vecs_rowspace_nonlex = []

    
    layer2ttype2vecs_rowspace = defaultdict(dict)
    layer2type2vecs_rowspace_nonlex = defaultdict(dict)
    layer2ttype2vecs_nullspaces = defaultdict(dict)
    layer2type2vecs_nullspace_nonlex = defaultdict(dict)
        
    for layer in layer2data.keys():
    
        type2vecs_rowspace = {}
        type2vecs_rowspace_nonlex = {}
        
        for rc_type in ["src", "src_by", "orc", "orc_by", "orrc", "orrc_by", "orrc_that", "prc", "prrc", "prrc_that", "all"]:
        
            if do_random_projection:
                w = np.random.rand(iters, 768) - 0.5
                P_rowspace = debias.get_rowspace_projection(w)
                assert np.allclose(P_rowspace.dot(P_rowspace) - P_rowspace, 0)
                
            else:
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
            #W = np.array(layer2ws[layer][rc_type][0]).squeeze(1)
            #print(W.shape)
            #W /= np.linalg.norm(W, keepdims = True, axis = 1)
            #print(lex_x[mask_lex].shape, W.shape)
            #exit()
            #Q = np.abs((lex_x[mask_lex].dot(W.T)))*W
            #print(Q.shape)
            #exit()
            #type2vecs_rowspace[rc_type] = np.abs((lex_x[mask_lex].dot(W)))*W
            #type2vecs_rowspace_nonlex[rc_type] = np.abs(nonlex_x[mask_nonlex].dot(W))*W            
            
        layer2ttype2vecs_rowspace[layer] = type2vecs_rowspace
        layer2type2vecs_rowspace_nonlex[layer] = type2vecs_rowspace_nonlex
        
    return layer2ttype2vecs_rowspace, layer2type2vecs_rowspace_nonlex
    
      


def calc_sims(layer2ttype2vecs_rowspace, layer2type2vecs_rowspace_nonlex, classifier, iters):

    
    type2ind = {d:i for i,d in enumerate(layer2type2vecs_rowspace_nonlex["0"].keys()) if d != "all"}
    ind2type = {i:d for d,i in type2ind.items()}
    
    layer2sims = dict()
    
    for layer in layer2ttype2vecs_rowspace.keys():
    
        type2vecs_rowspace, type2vecs_rowspace_nonlex =  layer2ttype2vecs_rowspace[layer], layer2type2vecs_rowspace_nonlex[layer]
        sims = np.zeros((10,10))
    
        from sklearn.metrics.pairwise import cosine_similarity

        for key, vecs in type2vecs_rowspace.items():
            for key2, vecs2 in type2vecs_rowspace_nonlex.items():

                if key == "all" or key2 == "all": continue
                
                sims2 = cosine_similarity(vecs, vecs2)
                sims[type2ind[key], type2ind[key2]] = np.mean(sims2) #mean1_normed.dot(mean2_normed.T)
                
        layer2sims[layer] = sims

    labels = [ind2type[i].upper().replace("_","-") for i in range(len(ind2type))]
    for layer, sims in layer2sims.items():
    
        df = pd.DataFrame(sims, index = labels, columns = labels)
        print("Layer {}".format(layer))

        print(df)
        print("========================================================")

        plt.figure(figsize = (11,8))
        sn.heatmap(df, annot=True, cmap = "YlGnBu", vmin = 0, vmax = 1)
        #plt.title("Cosine similarity in RC subspace between various RCs. {}. classifier: {}. {}".format(layer,classifier, iters))
        #plt.show()
        plt.savefig("../results/plots/rowspace-similarity.layer={}.classifier={}.iters={}.random_projection={}.png".format(layer, classifier, iters, do_random_projection), dpi=300)










parser = argparse.ArgumentParser(description='test influence of INLP on agreement prediction',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--classifier', dest='classifier', type=str,
                        default="sgd-log")
parser.add_argument('--iters', dest='iters', type=int,
                        default=8)                    
parser.add_argument('--random-projection', dest='random_projection', type=int,
                        default=0)                                                                               
args = parser.parse_args()
do_random_projection = args.random_projection == 1            
layer2ttype2vecs_rowspace, layer2type2vecs_rowspace_nonlex = collect_vecs(args.classifier, args.iters, do_random_projection)        
calc_sims(layer2ttype2vecs_rowspace, layer2type2vecs_rowspace_nonlex, args.classifier, args.iters)   
    
    
    
