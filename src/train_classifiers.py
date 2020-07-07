
import argparse
import pickle
import numpy as np
import sklearn
from sklearn import linear_model, svm
import tqdm
from collections import defaultdict
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def load_data(path):

    with open(path, "rb") as f:

        return pickle.load(f)

def train_classifiers(train_dev_datasets, test_datasets, classifier, num_clfs = 1):

    if classifier == "svm":
        clf_class = sklearn.svm.LinearSVC
    elif classifier == "sgd":
        clf_class = sklearn.linear_model.SGDClassifier
    
    results = np.zeros((len(train_dev_datasets), len(train_dev_datasets)))
    type2ind = {d:i for i,d in enumerate(train_dev_datasets.keys())}
    ind2type = {i:d for d,i in type2ind.items()}

    for positive_type in tqdm.tqdm(train_dev_datasets.keys()):

        train_x, train_y = train_dev_datasets[positive_type]["train"]
        dev_x, dev_y = train_dev_datasets[positive_type]["dev"]

        accs = []
        accs_nolexoverlap = defaultdict(list)
        for i in range(num_clfs):
            clf = clf_class(tol=1e-6, max_iter = 5000)
            clf.fit(train_x, train_y)
            score_dev = clf.score(dev_x, dev_y)
            accs.append(score_dev)

            #print("\nPositive type: {}. dev acc (lexically overlapped, same positive type): {}".format(positive_type, np.mean(accs)))

            for positive_type2_nonlex in test_datasets.keys():
                nolexoverlap_x, nolexoverlap_y = test_datasets[positive_type2_nonlex]["train"]
                relevant = nolexoverlap_y != 0  # only positives
                nolexoverlap_x = nolexoverlap_x[relevant]
                nolexoverlap_y = nolexoverlap_y[relevant]
                score_nonlexoverlap = clf.score(nolexoverlap_x, nolexoverlap_y)
                accs_nolexoverlap[positive_type2_nonlex].append(score_nonlexoverlap)

        for positive_type2_nonlex in test_datasets.keys():

            mean, std = np.mean(accs_nolexoverlap[positive_type2_nonlex]), np.std(accs_nolexoverlap[positive_type2_nonlex])
            results[type2ind[positive_type], type2ind[positive_type2_nonlex]] = mean
            print("\t\t non-lexically-overlapped positive type: {}. Recall: {}. STD: {}".format(positive_type2_nonlex, mean, std))

    labels = [ind2type[i] for i in range(len(ind2type))]
    return labels, results
    
def plot(labels, results, layer, classifier, test_group):
    labels = [l.upper() for l in labels]
    
    df_cm = pd.DataFrame(results, index = labels,
                  columns = labels)
    print(df_cm)
    exit()
    
    plt.figure(figsize = (10,7))
    g = sn.heatmap(df_cm, annot=True,  annot_kws={"fontsize":16})
    sn.set(font_scale=5.0)
    plt.title("Recall on test positives (columns) for each training positive type (rows). {}. classifier: {}.\n Test group: {}".format(layer, classifier, test_group), fontsize = 12)
    #plt.show()
    plt.savefig("../results/plots/recall-pairs-{}-classifier:{}-test-group:{}.png".format(layer, classifier, test_group), dpi=300)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='collect training datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train-dev-path', dest='train_dev_path', type=str,
                        default="../data/datasets.5000a.layer=6.masked=True.pickle",
                        help='input_path')
    parser.add_argument('--test-path', dest='test_path', type=str,
                        default="../data/datasets.5000a.layer=6.masked=True.pickle",
                        help='input_path for the non-lexically-overlapped datta')
    parser.add_argument('--classifier', dest='classifier', type=str,
                        default="svm",
                        help='sgd/svm')
                            
    args = parser.parse_args()
    test_group = args.test_path.split(".")[-4]
    print(test_group)
    layer = "layer="+str(args.train_dev_path.split(".")[-3].split("=")[-1])
    if layer == "layer=-1": layer = "layer=12"
    print(layer)
    
    train_dev_datasets = load_data(args.train_dev_path)
    test_datasets = load_data(args.test_path)
    labels, results = train_classifiers(train_dev_datasets, test_datasets, args.classifier)
    plot(labels, results, layer, args.classifier, test_group)
