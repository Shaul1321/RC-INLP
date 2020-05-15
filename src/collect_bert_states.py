import bert
import argparse
from typing import List, Tuple
import pickle
import tqdm

def collect_bert_states(bert, data: List[Tuple], layers: List[int], final_transform: bool, mask_prob: float):
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
    parser.add_argument('--mask', dest='mask', type=bool,
                    default = True,
                    help='whether to mask the token over which the state is collected')
                    
    args = parser.parse_args()
    model = bert.BertEncoder(args.device)

    with open("../data/data.pickle", "rb") as f:
        data = pickle.load(f)

    mask_prob = 1.0 if args.mask else 0.0
    data_with_states = collect_bert_states(model, data, [args.layer], False, mask_prob)
    with open("../data/data_with_states.layer={}.masked={}.pickle".format(args.layer, args.mask), "wb") as f:
        pickle.dump(data_with_states, f)
