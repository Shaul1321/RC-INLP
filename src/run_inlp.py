
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

def run_inlp(train_dev_datasets, classifier, num_classifiers, run_on_all):

    type2proj = {}
    type2dir = {}
    
    print("Keys: {}".format(train_dev_datasets.keys()))
    alpha = 1 * 1e-4
    early_stopping = True
    
    
    if classifier == "sgd-log":
            clf = sklearn.linear_model.SGDClassifier
            #params = {"max_iter": 1250, "early_stopping": early_stopping, "random_state": 0, "n_jobs": 8, "loss": "log", "fit_intercept": True, "alpha": alpha,
            #"n_iter_no_change": 10, "eta0": 0.1, "learning_rate": "adaptive"}
            params = {"early_stopping": True, "eta0": 0.1, "learning_rate": "adaptive", "fit_intercept": False}
            
    if classifier == "sgd-hinge":
            clf = sklearn.linear_model.SGDClassifier
            #params = {"early_stopping": early_stopping, "random_state": 1, "n_jobs": 8, "loss": "hinge", "alpha": alpha}
            params = {"early_stopping": True, "eta0": 0.1, "learning_rate": "adaptive", "fit_intercept": False}
                        
    if classifier == "sgd-perceptron":
            clf = sklearn.linear_model.SGDClassifier
            params = {"early_stopping": early_stopping, "random_state": 0, "n_jobs": 8, "loss": "perceptron", "alpha": alpha}
            
    elif classifier == "svm":
            clf = sklearn.svm.LinearSVC
            params = {"max_iter": 1000, "dual": False, "random_state": 0}
                    
    # individual types of RCs
    
    for positive_type in tqdm.tqdm(train_dev_datasets.keys()):

            train_x, train_y, train_is_rc = train_dev_datasets[positive_type]["train"]
            dev_x, dev_y, dev_is_rc = train_dev_datasets[positive_type]["dev"]
            P, rowspace_projections, Ws, accs = debias.get_debiasing_projection(clf, params, num_classifiers, 768, True,
            0, train_x, train_y, dev_x, dev_y, by_class = False, Y_train_main = False, Y_dev_main = False, dropout_rate = 0)
            print("norms:", [np.linalg.norm(w) for w in Ws])
            print("accs:", accs)
            #print("orthogonality test:\n", np.array(Ws).squeeze(1).dot(np.array(Ws).squeeze(1).T))
            print("==============================================================")
            
            type2proj[positive_type] = P
            type2dir[positive_type] = (Ws,accs)
            
    # all RCs
    """
    train_x, train_y = np.concatenate([train_dev_datasets[positive_type]["train"][0] for positive_type in train_dev_datasets.keys()], axis = 0), np.concatenate([train_dev_datasets[positive_type]["train"][1] for positive_type in train_dev_datasets.keys()], axis = 0)
    dev_x, dev_y = np.concatenate([train_dev_datasets[positive_type]["dev"][0] for positive_type in train_dev_datasets.keys()], axis = 0), np.concatenate([train_dev_datasets[positive_type]["dev"][1] for positive_type in train_dev_datasets.keys()], axis = 0)
            
    P, rowspace_projections, Ws = debias.get_debiasing_projection(clf, params, num_classifiers, 768, True,
    0, train_x, train_y, dev_x, dev_y, by_class = False, Y_train_main = False, Y_dev_main = False, dropout_rate = 0)
    type2proj["all"] = P                
    """
                     
    return type2proj, type2dir
    
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
                        default="../data/datasets.adapt.layer=6.masked=False.model=bert.pickle",
                        help='input_path')
    parser.add_argument('--classifier', dest='classifier', type=str,
                        default="svm",
                        help='sgd/svm')
    parser.add_argument('--num-classifiers', dest='num_classifiers', type=int,
                        default=8,
                        help='number of inlp classifiers')
    parser.add_argument('--all', dest='all', type=int,
                        default=0,
                        help='wehther to run on all RC vs. non-RC')
                                                    
    args = parser.parse_args()
    run_on_all = args.all == 1
    
    layer = "layer="+str(args.train_dev_path.split(".")[-4].split("=")[-1])
    masked = "masked="+str(args.train_dev_path.split(".")[-3].split("=")[-1])

    if layer == "layer=-1": layer = "layer=12"
    
    train_dev_datasets = load_data(args.train_dev_path)
    print(layer)
    type2proj, type2dir = run_inlp(train_dev_datasets, args.classifier, args.num_classifiers, run_on_all)
    
    with open("../data/type2P.{}.iters={}.classifier={}.{}.pickle".format(layer, args.num_classifiers, args.classifier, masked), "wb") as f:
    
        pickle.dump(type2proj, f)
        
    with open("../data/type2W.{}.iters={}.classifier={}.{}.pickle".format(layer, args.num_classifiers, args.classifier, masked), "wb") as f:
    
        pickle.dump(type2dir, f)
