import argparse
import pickle
import bert_agreement
import numpy as np
import copy
import torch
import os
from collections import defaultdict
import argparse
import random


def get_diff(str1, str2):
    str1_lst, str2_lst = str1.split(" "), str2.split(" ")
    i = [j for j in range(len(str1_lst)) if str1_lst[j] != str2_lst[j]]
    assert len(i) == 1
    i = i[0]
    return i, str1_lst[i], str2_lst[i]



def load_data(agreement_file_type, only_attractors = False, only_not_attractors = False):
    #with open("../marvin-linzen-data/{}.pickle".format(agreement_file_type), "rb") as f:
    with open(agreement_file_type, "rb") as f:
        raw_data = pickle.load(f)
    
    data = []
    i = 0
    
    for key in list(raw_data.keys())[:]:
     
     if "prc" in key or "prrc" in key: # fix format of the prc/prrc files
        
        new = key.replace("subj_s", "sing").replace("subj_p", "plur").replace("attractor_s", "sing").replace("attractor_p", "plur")
        raw_data[new] = raw_data[key]
        del raw_data[key]
    

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

            #correct_lst, wrong_lst = correct.split(" "), wrong.split(" ")
            verb_ind, correct_verb, wrong_verb = get_diff(correct, wrong)
     
            data.append({"sent": correct, "verb_index": verb_ind, "correct_verb": correct_verb,
                     "wrong_verb": wrong_verb})

                
    random.seed(0)
    random.shuffle(data)
   
    
    for i in range(3):
        print(data[i]["sent"], data[i]["correct_verb"], data[i]["wrong_verb"])
        print("=========================")
        
    return data


def get_accuracy(data_with_states):
    probs_good = []
    probs_bad = []

    good, bad = 0, 0
    for d in data_with_states:
        if (d["correct_word_prob"] is None) and d["wrong_word_prob"] is None:
        #    print(d["top_preds"][:10])
        #    print(d["sent"])
        #    print("================")
            continue
        if d["correct_word_prob"] is None and d["wrong_word_prob"] is not None:
        
            bad += 1
            probs_good.append(0)
            probs_bad.append(d["wrong_word_prob"]["prob"])
            continue
            
        if d["correct_word_prob"] is not None and d["wrong_word_prob"] is None:
        
            good += 1
            probs_good.append(d["correct_word_prob"]["prob"])
            probs_bad.append(0)
            continue
            
        
        #if d["correct_word_rank"] > 5000 and d["wrong_word_rank"] > 5000: continue
        
        rank_good, rank_bad = d["correct_word_prob"]["rank"], d["wrong_word_prob"]["rank"]
        prob_good, prob_bad = d["correct_word_prob"]["prob"], d["wrong_word_prob"]["prob"]
        probs_good.append(prob_good)
        probs_bad.append(prob_bad)

        if rank_good < rank_bad:
            good += 1
        else:
            bad += 1

    return good / (good + bad + 1e-6), np.mean(prob_good), np.mean(prob_bad)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='test influence of INLP on agreement prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

                        
    #parser.add_argument('--agreement-file-type', dest='agreement_file_type', type=str,
    #                    default="subj_rel",
    #                    help='subj_rel/obj_rel_across_inanim/simple_agrmt etc')
    parser.add_argument('--layer', dest='layer', type=int,
                        default=6,
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
                      
    parser.add_argument('--alpha', dest='alpha', type=float,
                        default=0,
                        help="whether to only perform nullspace projection (without counterfactual projection)")
    parser.add_argument('--classifier', dest='classifier', type=str,
                        default="sgd",
                        help="whether to only perform nullspace projection (without counterfactual projection)")
    parser.add_argument('--iters', dest='iters', type=int,
                        default=8,
                        help="whether to only perform nullspace projection (without counterfactual projection)")
                    
                                                                             
    args = parser.parse_args()
    only_attractors = args.only_attractors == 1
    only_not_attractors = args.only_not_attractors == 1
    alpha = args.alpha# == 0
    
    print("only attractors:", only_attractors, "only not attractors:", only_not_attractors, "alpha:", alpha)
    results = defaultdict(dict)
    
    
    with open("../data/type2P.layer={}.iters={}.classifier={}.masked=True.pickle".format(args.layer, args.iters, args.classifier), "rb") as f:
        type2P = pickle.load(f)   
    with open("../data/type2P.layer={}.iters={}.classifier={}.masked=True.pickle".format(args.layer, args.iters, args.classifier), "rb") as f:
        type2P2 = pickle.load(f) 
            
    relevant = ["subj_rel.pickle", "sent_comp.pickle", "obj_rel_across_anim.pickle", "obj_rel_across_inanim.pickle", "obj_rel_no_comp_across_anim.pickle", "obj_rel_no_comp_across_inanim.pickle", "obj_rel_no_comp_within_anim.pickle", "obj_rel_no_comp_within_inanim.pickle", "obj_rel_within_anim.pickle", "prc_anim.pickle", "prc_inanim.pickle", "prrc_anim.pickle", "prrc_inanim.pickle", "simple_agrmt.pickle"]
    
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
            n = 350
            data_with_states_before = bert_agreement.collect_bert_states(bert, copy.deepcopy(data[:n]), layer = args.layer, P = I, P2 = None, alpha = alpha)
            data_with_states_after = bert_agreement.collect_bert_states(bert, copy.deepcopy(data[:n]), layer = args.layer, P = P, P2 = P2, alpha = alpha)

            
            acc_before, prob_good_before, prob_bad_before = get_accuracy(data_with_states_before)
            acc_after, prob_good_after, prob_bad_after = get_accuracy(data_with_states_after)
            
            r =  {"acc_before": acc_before, "acc_after": acc_after, "prob_good_before": prob_good_before, "prob_bad_before": prob_bad_before, "prob_good_after": prob_good_after, "prob_bad_after": prob_bad_after}
            r["acc_before"]+=1e-5
            print("before", r["acc_before"], "after", r["acc_after"], "relative change", (r["acc_before"]-r["acc_after"])/r["acc_before"]*100)
            results[filename][P_type] =r
            print("=====================================================")
    
    fname = "../data/agreement_results.{}.layer={}.only_attractors={}.only_not_attractors={}.alpha={}.classifier={}.pickle".format(args.iters, args.layer, only_attractors, only_not_attractors, alpha, args.classifier)
    with open(fname, "wb") as f:
        print("Saving {}".format(fname))
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
    
