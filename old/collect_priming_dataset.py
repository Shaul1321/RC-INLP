import argparse
import urllib
import pickle
import spacy
import os
import numpy as np
import random
from sklearn.utils import shuffle

random.seed(0)
np.random.seed(0)

pos2neg_fnames = {"src": "scont", "src_by": "scont_by", "orc": "ocont", "orc_by": "ocont_by", "orrc": "ocont", "orrc_by": "ocont_by", "orrc_that": "ocont_that", "prc": "ocont", "prrc": "ocont",  "prrc_that": "ocont_that"}


def collect_data(sentences_per_type, ignore_prob, path = "../rc_dataset_new"):

    data = []
    sents_collected = set()

    for sentences_set in os.listdir(path):

        for filename in os.listdir(path + "/" + sentences_set):
            type_sents = filename.split("_", 1)[1].split(".")[0]
            print(type_sents)
            
            if type_sents in pos2neg_fnames.keys():
                pos_path = path + "/" + sentences_set + "/" + filename
                neg_type_sents = pos2neg_fnames[type_sents]
                neg_path = path + "/" + sentences_set + "/" + filename.split("_")[0]+"_"+neg_type_sents+".txt"
                
                with open(pos_path, "r", encoding="utf-8") as f:

                    lines_pos = f.readlines()
                with open(neg_path, "r", encoding="utf-8") as f:

                    lines_neg = f.readlines()

                lines_pos, lines_neg = shuffle(lines_pos[1:], lines_neg[1:], random_state = 0)
                lines_pos, lines_neg = lines_pos[:sentences_per_type], lines_neg[:sentences_per_type]
                
                for i, (pos_line, neg_line) in enumerate(zip(lines_pos, lines_neg)):

                    txt_pos, start_pos, length_pos, plurality_pos = pos_line.strip().split(",")
                    txt_neg, start_neg, length_neg, plurality_neg = neg_line.strip().split(",")
                    start, length = int(start_pos), int(length_pos)
                    end = start + length # end is one index post the end of the RC.
                    
                    # positives
                    idx = np.random.choice(range(start, end), size = 3, replace = False)
                    for relc_index in idx:
                        #if np.random.random() < ignore_prob: continue
                        
                        data.append({"text": txt_pos, "start": start, "end": end, "label": 1, "ind": relc_index,
                                     "sent_type": type_sents, "sentences_set": sentences_set})
                        w = txt_pos.split(" ")[relc_index]

                        # negatives
                        #if txt_neg in sents_collected: continue
                        found_negative = False
                        
                        if np.random.random() < 1.0/3:
                            
                            # choose a "corresponding" word (same lexical item) from a non-RC sentence as negative

                            options = []
                            
                            for ind, w2 in enumerate(txt_neg.split(" ")):
                                if w2 == w:
                                    options.append((ind, w2))
                                    found_negative = True
                            
                            if found_negative:
                                ind, w2 = random.choice(options)
                                data.append({"text": txt_neg, "ind": ind, "label": 0, "sent_type": neg_type_sents,
                                    "sentences_set": sentences_set, "type_neg": "non-rc-corresponding"})
                                        
                        if not found_negative and np.random.random() < 0.5:
                            
                            # choose a random word from a non-RC sentence as negative
                            
                            splitted_neg = txt_neg.split(" ")
                            ind = np.random.choice(range(len(splitted_neg)))
                            w2 = splitted_neg[ind]
                            data.append({"text": txt_neg, "ind": ind, "label": 0, "sent_type": neg_type_sents,
                                    "sentences_set": sentences_set, "type_neg": "non-rc-random"})
                            found_negative =  True
                            
                        if not found_negative:
                        
                            # choose a word from a RC sentence (but one that is outside the RC) as a negative
                            # choose previous or consecutive word in 50% acc
                            
                            splitted_pos = txt_pos.split(" ")
                            
                            if np.random.random() < 0.5:
                            
                                ind = np.random.choice(range(end, len(splitted_pos)))
                               
                            else:
                                ind = np.random.choice(range(0, start))
                            
                            w2 = splitted_pos[ind]
                            data.append({"text": txt_pos, "ind": ind, "label": 0, "sent_type": type_sents,
                                    "sentences_set": sentences_set, "type_neg": "outside-rc"})
                            
                        #sents_collected.add(txt_neg)
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description='Collecting priming-data RC dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--sentences_per_type', dest='sentences_per_type', type=int,
                    default=1000,
                    help='how many sentences to take from each file.')
    parser.add_argument('--ignore_prob', dest='ignore_prob', type=float,
                    default=0.6,
                    help='probability to skip an index within a RC')
                    
    args = parser.parse_args()
    data = collect_data(args.sentences_per_type, args.ignore_prob)
    with open("../data/data.pickle", "wb") as f:

        pickle.dump(data, f)

    print("Collected {} examples".format(len(data)))
    pos = [d for d in data if d["label"] == 1]
    neg = [d for d in data if d["label"] == 0]
    print("Proportion positive: {}".format(len(pos)/(len(pos) + len(neg))  ))
    
    for typ in ["src", "src_by", "orc", "orc_by", "orrc", "orrc_by", "orrc_that", "prc", "prrc", "prrc_that"]:
    
        relevant = [d for d in data if d["sent_type"] == typ]
        print(typ, len(relevant))
