import argparse
import urllib
import pickle
import spacy
import os

pos2neg_fnames = {"src": "scont", "orc": "ocont", "orrc": "ocont", "prc": "ocont", "prrc": "ocont"}

def collect_data(sentences_per_type, path = "../rc_dataset"):

    data = []

    for sentences_set in os.listdir(path):
        for filename in os.listdir(path + "/" + sentences_set):
            type_sents = filename.split("_")[1].split(".")[0]
            if type_sents in pos2neg_fnames.keys():
                pos_path = path + "/" + sentences_set + "/" + filename
                neg_type_sents = pos2neg_fnames[type_sents]
                neg_path = path + "/" + sentences_set + "/" + filename.split("_")[0]+"_"+neg_type_sents+".txt"

                with open(pos_path, "r", encoding="utf-8") as f:

                    lines_pos = f.readlines()[:sentences_per_type]
                with open(neg_path, "r", encoding="utf-8") as f:

                    lines_neg = f.readlines()[:sentences_per_type]

                # collect instances of positive and negative examples

                for pos_line, neg_line in zip(lines_pos[1:], lines_neg[1:]):

                    txt_pos, start_pos, length_pos = pos_line.strip().split(",")
                    txt_neg, start_neg, length_neg = neg_line.strip().split(",")
                    start, length = int(start_pos), int(length_pos)
                    end = start + length

                    for relc_index in range(start, end):

                        data.append({"text": txt_pos, "start": start, "end": end, "label": 1, "ind": relc_index,
                                     "sent_type": type_sents, "sentences_set": sentences_set})
                        w = txt_pos.split(" ")[relc_index]
                        for ind, w2 in enumerate(txt_neg.split(" ")):
                            if w2 == w:
                                data.append({"text": txt_neg, "ind": ind, "label": 0, "sent_type": neg_type_sents,
                                             "sentences_set": sentences_set})

    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description='Collecting priming-data RC dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--sentences_per_type', dest='sentences_per_type', type=int,
                    default=400,
                    help='how many sentences to take from each file.')
    args = parser.parse_args()
    data = collect_data(args.sentences_per_type)
    with open("../data/data.pickle", "wb") as f:

        pickle.dump(data, f)

    print("Collected {} examples".format(len(data)))