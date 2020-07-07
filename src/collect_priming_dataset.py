import argparse
import urllib
import pickle
import spacy
import os
import numpy as np
import random

pos2neg_fnames = {"src": "scont", "orc": "ocont", "orrc": "ocont", "prc": "ocont", "prrc": "ocont"}

def collect_data(sentences_per_type, path = "../rc_dataset"):

    data = []
    sents_collected = set()

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

                ## subsample sentences
                #lines_pos = lines_pos[:sentences_per_type]
                #if sentences_set == "5000a" or (type_sents in pos2neg_fnames.keys()):
                #    lines_neg = lines_neg[:sentences_per_type]

                # collect instances of positive and negative examples

                for i, (pos_line, neg_line) in enumerate(zip(lines_pos[1:], lines_neg[1:])):

                    txt_pos, start_pos, length_pos = pos_line.strip().split(",")
                    txt_neg, start_neg, length_neg = neg_line.strip().split(",")
                    start, length = int(start_pos), int(length_pos)
                    end = start + length # end is one index post the end of the RC.

                    # positives

                    for relc_index in range(start, end):

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
                    default=300,
                    help='how many sentences to take from each file.')
    args = parser.parse_args()
    data = collect_data(args.sentences_per_type)
    with open("../data/data.pickle", "wb") as f:

        pickle.dump(data, f)

    print("Collected {} examples".format(len(data)))
