import bert
import argparse
from typing import List, Tuple
import pickle
import tqdm

def collect_bert_states(bert, data: List[Tuple], layers: List[int], final_transform: bool, mask_prob: float, use_roberta):
    data_with_states = []

    for i, d in tqdm.tqdm(enumerate(data), total=len(data)):
        sent, ind, label = d["text"], d["ind"], d["label"]

        preds, orig2tok, bert_tokens = bert.encode(sent, layers=layers, pos_ind=ind,
                                                   mask_prob=mask_prob, final_transform=final_transform)

        j = orig2tok[ind]
        v = preds[j]

        dict_with_state = d.copy()
        dict_with_state["vec"] = v
        data_with_states.append(dict_with_state)

    return data_with_states

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description='Collecting priming-data RC dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--device', dest='device', type=str,
                    default="cuda",
                    help='cpu/cuda')
    parser.add_argument('--layer', dest='layer', type=int,
                    default = 6,
                    help='layer to take')
    parser.add_argument('--mask', dest='mask', type=int,
                    default =1,
                    help='whether to mask the token over which the state is collected')
    parser.add_argument('--random', dest='random', type=int,
                    default =0,
                    help='whether to use randomly initialized bert')                    
    parser.add_argument('--label', dest='label', type=str,
                    default ="",
                    help='label string to concatenate to layer name')
    parser.add_argument('--use-roberta', dest='use_roberta', type=int,
                    default =0,
                    help='label string to concatenate to layer name')
                                            
    args = parser.parse_args()
    args.random = True if args.random == 1 else False
    layer = args.layer
    use_roberta = args.use_roberta == 1
    
    if args.random:
        args.layer = str(args.layer) + "-random"
    print(args.mask)
    print(args.layer)
    print(args.random)
    model = bert.BertEncoder(args.device, random = args.random, use_roberta = use_roberta)

    with open("../data/data.pickle", "rb") as f:
        data = pickle.load(f)

    mask_prob = args.mask
    data_with_states = collect_bert_states(model, data, [layer], False, mask_prob, use_roberta)
    masked_str = "True" if args.mask == 1 else "False"
    model_name = "bert" if not use_roberta else "roberta"
    fname = "../data/data_with_states.layer={}.masked={}.model={}.pickle".format(str(args.layer) + args.label, masked_str, model_name)
    with open(fname, "wb") as f:
        print("Saving {}".format(fname))
        pickle.dump(data_with_states, f)
