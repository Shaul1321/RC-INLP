import argparse
import pickle
import bert_agreement
import numpy as np
import copy
import torch
import os
from collections import defaultdict
import argparse

def load_data(agreement_file_type, only_attractors = False, only_not_attractors = False):
    #with open("../marvin-linzen-data/{}.pickle".format(agreement_file_type), "rb") as f:
    with open(agreement_file_type, "rb") as f:
        raw_data = pickle.load(f)
    
    data = []
    i = 0
    print("=====================================================")
    print(agreement_file_type)
    print("keys:", raw_data.keys())
    
    for key in raw_data.keys():

        if not ("sent_comp" in agreement_file_type or "simple_agrmt" in agreement_file_type): # = if has agreement across RC
        
            if only_attractors:
                if key.split("_").count("sing") == 2 or key.split("_").count("plur") == 2: # if no attractors

                        continue # skip since there are no attractors
            elif only_not_attractors:
                print(key.split("_"))
                if not (key.split("_").count("sing") == 2 or key.split("_").count("plur") == 2): # if attractors

                    continue
                
        examples = raw_data[key]
        for correct, wrong in examples:
            i += 1
            correct += " ."
            wrong += " ."

            correct_lst, wrong_lst = correct.split(" "), wrong.split(" ")
            correct_verb, wrong_verb = correct_lst[-3], wrong_lst[-3]
            data.append({"sent": correct, "verb_index": len(correct_lst) - 3, "correct_verb": correct_verb,
                     "wrong_verb": wrong_verb}) #note: verb_index is actually verb_index + 1 (handling the period) 
                     
            #if i == 1:
            #    print(data)
            
    return data


def get_accuracy(data_with_states):
    probs_good = []
    probs_bad = []

    good, bad = 0, 0
    for d in data_with_states:
        if d["correct_word_prob"] is None or d["wrong_word_prob"] is None:
        #    print(d["top_preds"][:10])
        #    print(d["sent"])
        #    print("================")
            continue
        rank_good, rank_bad = d["correct_word_prob"]["rank"], d["wrong_word_prob"]["rank"]
        prob_good, prob_bad = d["correct_word_prob"]["prob"], d["wrong_word_prob"]["prob"]
        probs_good.append(prob_good)
        probs_bad.append(prob_bad)

        if rank_good < rank_bad:
            good += 1
        else:
            bad += 1

    return good / (good + bad), np.mean(prob_good), np.mean(prob_bad)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='test influence of INLP on agreement prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--P-path', dest='P_path', type=str,
                        default="../data/type2P.layer=3.iters=16.classifier=svm.masked=True.pickle",
                        help='input_path')
                        
    parser.add_argument('--P-path2', dest='P_path2', type=str,
                        default="../data/type2P.layer=3.iters=16.classifier=svm.masked=True.pickle",
                        help='input_path')
                        
    #parser.add_argument('--agreement-file-type', dest='agreement_file_type', type=str,
    #                    default="subj_rel",
    #                    help='subj_rel/obj_rel_across_inanim/simple_agrmt etc')
    parser.add_argument('--layer', dest='layer', type=int,
                        default=3,
                        help='bert layer')
    parser.add_argument('--device', dest='device', type=str,
                        default="cuda",
                        help="cpu/cuda")
    parser.add_argument('--only-attractors', dest='only_attractors', type=int,
                        default=0,
                        help="whether to keep only sentences with attractors among the RC sentences")
    parser.add_argument('--only-not-attractors', dest='only_not_attractors', type=int,
                        default=0,
                        help="whether to keep only sentences without attractors among the RC sentences")
                        
                        
    args = parser.parse_args()
    only_attractors = args.only_attractors == 1
    only_not_attractors = args.only_not_attractors == 1
    print("only attractors:", only_attractors, "only not attractors:", only_not_attractors)
    results = defaultdict(dict)
    
    with open(args.P_path, "rb") as f:
        type2P = pickle.load(f)   
    with open(args.P_path2, "rb") as f:
        type2P2 = pickle.load(f) 
            
    relevant = ["subj_rel.pickle", "sent_comp.pickle", "obj_rel_across_anim.pickle", "obj_rel_across_inanim.pickle", "obj_rel_no_comp_across_anim.pickle", "obj_rel_no_comp_across_inanim.pickle", "obj_rel_no_comp_within_anim.pickle", "obj_rel_no_comp_within_inanim.pickle", "obj_rel_within_anim.pickle", "simple_agrmt.pickle"]
    
    #relevant = ["subj_rel.pickle", "sent_comp.pickle", "obj_rel_no_comp_across_inanim.pickle"]
    
    for filename in os.listdir("../marvin-linzen-data/"):
    
        if filename not in relevant: continue
        if "within" in filename: continue
        
        for P_type in type2P.keys():

            print(P_type, filename)
            
            P = torch.tensor(type2P[P_type]).float().to(args.device)
            P2 = torch.tensor(type2P2[P_type]).float().to(args.device)
            I = torch.eye(P.shape[0]).float().to(args.device)

            data = load_data("../marvin-linzen-data/" + filename, only_attractors, only_not_attractors)
            bert = bert_agreement.BertEncoder(args.device)
            n = 250
            data_with_states_before = bert_agreement.collect_bert_states(bert, copy.deepcopy(data[:n]), layer = args.layer, P = P, P2 = None)
            data_with_states_after = bert_agreement.collect_bert_states(bert, copy.deepcopy(data[:n]), layer = args.layer, P = P, P2 = P2)

            acc_before, prob_good_before, prob_bad_before = get_accuracy(data_with_states_before)
            acc_after, prob_good_after, prob_bad_after = get_accuracy(data_with_states_after)
            
            r =  {"acc_before": acc_before, "acc_after": acc_after, "prob_good_before": prob_good_before, "prob_bad_before": prob_bad_before, "prob_good_after": prob_good_after, "prob_bad_after": prob_bad_after}
            print("before", r["acc_before"], "after", r["acc_after"], "relative change", (r["acc_before"]-r["acc_after"])/r["acc_before"]*100)
            results[filename][P_type] =r
            
    
    with open("../data/agreement_results.16.layer={}.only_attractors={}.only_not_attractors={}.pickle".format(args.layer, only_attractors, only_not_attractors), "wb") as f:
        pickle.dump(results, f)
                
    """ 
    print("acc before:", acc_before, "acc after:", acc_after)
    print("prob_good_before:", prob_good_before, "prob_good_after:", prob_good_after)
    print("prob_bad_before:", prob_bad_before, "prob_bad_after:", prob_bad_after)


    print("==========================================================================")

    for i in range(25):
        if data_with_states_before[i]["top_preds"][0] != data_with_states_after[i]["top_preds"][0]:
            print(data_with_states_before[i]["sent"])
            print("Predictions before: {}".format(data_with_states_before[i]["top_preds"][:15]))
            print("Predictions after: {}".format(data_with_states_after[i]["top_preds"][:15]))
            print("=============================================================")
    """
    
