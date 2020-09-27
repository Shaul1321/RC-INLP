import bert
import argparse
from typing import List, Tuple
import pickle
import tqdm
import sys

sys.path.append("inlp/")
from inlp import debias
from sklearn.utils import shuffle
import random
import numpy as np
from collections import defaultdict

pos2neg_fnames = {"src": "scont", "src_by": "scont_by", "orc": "ocont", "orc_by": "ocont_by", "orrc": "ocont", "orrc_by": "ocont_by", "orrc_that": "ocont_that", "prc": "ocont", "prrc": "ocont",  "prrc_that": "ocont_that"}


def construct_dataset(examples: dict):
    x, y, is_rc = [], [], []
    for d in examples:
        x.append(d["vec"])
        y.append(d["label"])
        is_rc.append(d["is_rc"])

    x = np.array(x)
    y = np.array(y)
    is_rc = np.array(is_rc)
    
    return x, y, is_rc


def remove_duplicates(examples: dict):
    without_duplicates = []
    keys_set = set()

    for d in examples:

        txt, ind = d["text"], d["ind"]
        key = txt + "-" + str(ind)
        if key not in keys_set:
            without_duplicates.append(d)
            keys_set.add(key)
    return without_duplicates


def get_train_dev_test(sentences_group: str, input_path: str):
    with open(input_path, "rb") as f:

        all_data = pickle.load(f)

    sent_type2data = defaultdict(dict)

    positive_examples = [d for d in all_data if d["sentences_set"] == sentences_group]
    negative_examples = [d for d in all_data if d["sentences_set"] == sentences_group]
    
    # remove duplicates

    #negative_examples = remove_duplicates(negative_examples)
    #print(sentences_group, len(positive_examples)/len(negative_examples))
    
    print("========================================================================")

    # collect exampler per type

    for pos_sent_type in pos2neg_fnames.keys():
        neg_sent_type = pos2neg_fnames[pos_sent_type]
        positive_examples_relevant = [d for d in positive_examples if d["sent_type"] == pos_sent_type]
        negative_examples_relevant = [d for d in negative_examples if d["sent_type"] == neg_sent_type]
        all = positive_examples_relevant + negative_examples_relevant
        pos, neg = [d for d in all if d["label"] == 1], [d for d in all if d["label"] == 0]
        
        
        m = min(len(pos), len(neg)) # ensure equal number of pos and neg examples
        pos, neg = pos[:m], neg[:m]
        
        #print(pos_sent_type, neg_sent_type, len(positive_examples_relevant), len(negative_examples_relevant),  len(positive_examples_relevant)/len(negative_examples_relevant))
        all_relevant = pos + neg
       
        for k,d in enumerate(all_relevant):
            is_rc = "and" not in d["text"]
            all_relevant[k]["is_rc"] = is_rc   
        
        random.seed(0)
        random.shuffle(all_relevant)

        train_len = int(0.8 * len(all_relevant))
        train_examples_relevant, dev_examples_relevant = all_relevant[:train_len], all_relevant[train_len:]

        train_x, train_y, train_is_rc = construct_dataset(train_examples_relevant)
        dev_x, dev_y, dev_is_rc = construct_dataset(dev_examples_relevant)

        sent_type2data[pos_sent_type]["train"] = (train_x, train_y, train_is_rc)
        sent_type2data[pos_sent_type]["dev"] = (dev_x, dev_y, dev_is_rc)
        # sent_type2data[pos_sent_type]["test"] = (test_x, test_y)
        print(pos_sent_type, neg_sent_type, train_x.shape, dev_x.shape, dev_y.sum() / len(dev_y))


    # concatenate to also have data over all rc types
    
    train_x = np.concatenate([sent_type2data[positive_type]["train"][0] for positive_type in sent_type2data.keys()], axis = 0)
    train_y = np.concatenate([sent_type2data[positive_type]["train"][1] for positive_type in sent_type2data.keys()], axis = 0)
    train_is_rc = np.concatenate([sent_type2data[positive_type]["train"][2] for positive_type in sent_type2data.keys()], axis = 0)

    dev_x = np.concatenate([sent_type2data[positive_type]["dev"][0] for positive_type in sent_type2data.keys()], axis = 0)
    dev_y = np.concatenate([sent_type2data[positive_type]["dev"][1] for positive_type in sent_type2data.keys()], axis = 0)
    dev_is_rc = np.concatenate([sent_type2data[positive_type]["dev"][2] for positive_type in sent_type2data.keys()], axis = 0)
    
    sent_type2data["all"]["train"] = (train_x, train_y, train_is_rc)
    sent_type2data["all"]["dev"] = (dev_x, dev_y, dev_is_rc)
    
    return sent_type2data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='collect training datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input-path', dest='input_path', type=str,
                        default="../data/data_with_states.layer=6.masked=True.model=roberta.pickle",
                        help='input_path')
    parser.add_argument('--sentences-group', dest='sentences_group', type=str,
                        default='adapt',
                        help='adapt/test')

    args = parser.parse_args()
    layer = args.input_path.split(".")[-4]
    masked = args.input_path.split(".")[-3]
    model = args.input_path.split(".")[-2]
    print("model is {}".format(model))
    print("layer is {}".format(layer))
    sent_type2data = get_train_dev_test(args.sentences_group, args.input_path)

    fname = "../data/datasets.{}.{}.{}.{}.pickle".format(args.sentences_group, layer, masked, model)
    with open(fname, "wb") as f:
        print("Saving {}".format(fname))
        pickle.dump(sent_type2data, f)
