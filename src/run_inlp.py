
import argparse
import pickle
import numpy as np
import sklearn
from sklearn import linear_model, svm
import tqdm
from collections import defaultdict
import seaborn as sn
import pandas as pd
import sys
sys.path.append("inlp/")
from inlp import debias
import matplotlib.pyplot as plt

def load_data(path):

    with open(path, "rb") as f:

        return pickle.load(f)

def run_inlp(train_dev_datasets, classifier, num_classifiers):

    type2proj = {}

    for positive_type in tqdm.tqdm(train_dev_datasets.keys()):

        train_x, train_y = train_dev_datasets[positive_type]["train"]
        dev_x, dev_y = train_dev_datasets[positive_type]["dev"]

        if classifier == "sgd":
            clf = sklearn.linear_model.SGDClassifier
            params = {"tol":1e-6, "max_iter": 2500}
        elif classifier == "svm":
             clf = sklearn.svm.LinearSVC
             params = {"tol":1e-6, "max_iter": 7000}
                        
        P, rowspace_projections, Ws = debias.get_debiasing_projection(clf, params, num_classifiers, 768, True,
        0, train_x, train_y, dev_x, dev_y, by_class = False, Y_train_main = False, Y_dev_main = False, dropout_rate = 0)
        
        type2proj[positive_type] = P
    
    return type2proj
    
def plot(labels, results, layer):

    df_cm = pd.DataFrame(results, index = labels,
                  columns = labels)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.title("Recall on test positives (columns) for each training positive type (rows). {}".format(layer))
    #plt.show()
    plt.savefig("../results/plots/recall-pairs-{}.png".format(layer), dpi=600)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='collect training datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train-dev-path', dest='train_dev_path', type=str,
                        default="../data/datasets.5000a.layer=layer=6.pickle",
                        help='input_path')
    parser.add_argument('--classifier', dest='classifier', type=str,
                        default="svm",
                        help='sgd/svm')
    parser.add_argument('--num-classifiers', dest='num_classifiers', type=int,
                        default=16,
                        help='number of inlp classifiers')
                            
    args = parser.parse_args()
    layer = "layer="+str(args.train_dev_path.split(".")[-2].split("=")[-1])
    if layer == "layer=-1": layer = "layer=12"
    print(layer)
    
    train_dev_datasets = load_data(args.train_dev_path)

    type2proj = run_inlp(train_dev_datasets, args.classifier, args.num_classifiers)
    with open("../data/type2P.{}.iters={}.classifier={}.pickle".format(layer, args.num_classifiers, args.classifier), "wb") as f:
    
        pickle.dump(type2proj, f)
