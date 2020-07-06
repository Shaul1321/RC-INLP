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

pos2neg_fnames = {"src": "scont", "orc": "ocont", "orrc": "ocont", "prc": "ocont", "prrc": "ocont"}


def construct_dataset(examples: dict):
    x, y = [], []
    for d in examples:
        x.append(d["vec"])
        y.append(d["label"])

    x = np.array(x)
    y = np.array(y)
    return x, y


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

    negative_examples = remove_duplicates(negative_examples)

    # collect exampler per type

    for pos_sent_type in pos2neg_fnames.keys():
        neg_sent_type = pos2neg_fnames[pos_sent_type]
        positive_examples_relevant = [d for d in positive_examples if d["sent_type"] == pos_sent_type]
        negative_examples_relevant = [d for d in negative_examples if d["sent_type"] == neg_sent_type]
        all_relevant = positive_examples_relevant + negative_examples_relevant

        random.seed(0)
        random.shuffle(all_relevant)

        train_len = int(0.8 * len(all_relevant))
        train_examples_relevant, dev_examples_relevant = all_relevant[:train_len], all_relevant[train_len:]

        train_x, train_y = construct_dataset(train_examples_relevant)
        dev_x, dev_y = construct_dataset(dev_examples_relevant)

        sent_type2data[pos_sent_type]["train"] = (train_x, train_y)
        sent_type2data[pos_sent_type]["dev"] = (dev_x, dev_y)
        # sent_type2data[pos_sent_type]["test"] = (test_x, test_y)
        print(pos_sent_type, neg_sent_type, train_x.shape, dev_x.shape, dev_y.sum() / len(dev_y))

    return sent_type2data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='collect training datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input-path', dest='input_path', type=str,
                        default="../data/data_with_states.layer=9.masked=True.pickle",
                        help='input_path')
    parser.add_argument('--inlp-iterations', dest='inlp_iterations', type=int,
                        default=32,
                        help='number of INLP iterations to perform')
    parser.add_argument('--sentences-group', dest='sentences_group', type=str,
                        default='5000t',
                        help='5000a/5000t')

    args = parser.parse_args()
    layer = args.input_path.split(".")[-3]
    masked = args.input_path.split(".")[-2]

    sent_type2data = get_train_dev_test(args.sentences_group, args.input_path)

    with open("../data/datasets.{}.{}.{}.pickle".format(args.sentences_group, layer, masked), "wb") as f:

        pickle.dump(sent_type2data, f)
