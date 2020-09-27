import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, RobertaModel, RobertaForMaskedLM, \
    RobertaTokenizer, RobertaConfig
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AlbertTokenizer, AlbertModel, AlbertConfig
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig
from collections import defaultdict
from typing import Tuple, Dict
from typing import List
import tqdm


def forward_from_specific_layer(model, layer_number: int, layer_representation: torch.Tensor):
    """
   :param model: a BertForMaskedLM model
   :param layer_representation: a torch tensor, dims: [1, seq length, 768]
   Return:
           states, a numpy array. dims: [#LAYERS - layer_number, seq length, 768]
           last_state_after_batch_norm: np array, after batch norm. dims: [seq_length, 768]
   """

    layers = model.bert.encoder.layer[layer_number:]
    layers.append(model.cls.predictions.transform)

    h = layer_representation
    states = []

    with torch.no_grad():
        for i, layer in enumerate(layers):
            h = layer(h)[0] if i != len(layers) - 1 else layer(h)
            states.append(h)

    # states[-1] = states[-1].unsqueeze(0)

    for i, s in enumerate(states):
        states[i] = s.detach().cpu().numpy()

    states = np.array(states)

    for x in states:
        assert len(x.shape) == 3

    return states


class BertEncoder(object):

    def __init__(self, device='cpu'):

        # config = BertConfig.from_pretrained("bert-large-uncased-whole-word-masking", output_hidden_states=True)
        # self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
        # self.model = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking', config = config)
        config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)

        # config = AlbertConfig.from_pretrained("albert-xlarge-v2", output_hidden_states=True)
        # self.tokenizer = AlbertTokenizer.from_pretrained("albert-xlarge-v2")
        # self.model = AlbertModel.from_pretrained("albert-xlarge-v2", config = config)
        # config = RobertaConfig.from_pretrained("roberta-large", output_hidden_states=True)
        # self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        # self.model = RobertaModel.from_pretrained('roberta-large', config = config)
        self.final_transform = self.model.cls.predictions.transform
        self.out_embed = self.model.cls.predictions.decoder.weight.detach().cpu().numpy()

        self.model.eval()
        self.model.to(device)
        self.device = device

    def tokenize(self, original_sentence: List[str]) -> Tuple[List[str], Dict[int, int]]:

        """
        Parameters
        ----------
        Returns
        -------
        bert_tokens: The sentence, tokenized by BERT tokenizer.
        orig_to_tok_map: An output dictionary consisting of a mapping (alignment) between indices in the original tokenized sentence, and indices in the sentence tokenized by the BERT tokenizer. See https://github.com/google-research/bert
        """

        bert_tokens = ["[CLS]"]
        orig_to_tok_map = {}
        has_subwords = False
        is_subword = []

        for i, w in enumerate(original_sentence):
            tokenized_w = self.tokenizer.tokenize(w)
            has_subwords = len(tokenized_w) > 1
            is_subword.append(has_subwords)
            bert_tokens.extend(tokenized_w)

            orig_to_tok_map[i] = len(bert_tokens) - 1

        bert_tokens.append("[SEP]")

        return (bert_tokens, orig_to_tok_map)

    def encode(self, sentence: str, mask_index: int, layer: int, P: np.ndarray, P2: np.ndarray, alpha, ws, project=False):

        tokenized_text, orig2tok = self.tokenize(sentence.split(" "))
        #print(tokenized_text, mask_index, orig2tok)
        #exit()
        
        mask_idx_bert = orig2tok[mask_index]
        tokenized_text[mask_idx_bert] = "[MASK]"

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor)
        rep_state = [outputs[1][i][0] for i in range(len(outputs[1]))]
        
        states = rep_state[layer]
        #norm_original = torch.norm(states, dim = 1, keepdim = True)
        
        P_R = torch.eye(768).cuda() - P
        if alpha >= -1e-4:
            alpha_prime = alpha + 1
        else:
            alpha_prime = alpha
        
        states_projected = states.clone()
        #states_projected[mask_idx_bert] = (states[mask_idx_bert] - (alpha_prime)*(states[mask_idx_bert]@P_R)).float()
        
         ### SELECTIVE-FLIPPING-START
        if project:
            states_projected[mask_idx_bert] = (states[mask_idx_bert] - states[mask_idx_bert]@P_R).float()  
                
            signs = torch.sign(states[mask_idx_bert]@ws.T)
            if np.random.random() < 1/80: 
                print(states[mask_idx_bert]@ws.T)
                print("=========================================")
            
            for r, (s,w) in enumerate(zip(signs, ws)):
                #if r > 4: continue
            
                proj = (states[mask_idx_bert]@w)*w
                alpha_sign = 1 if alpha > 0 else -1
                if alpha_sign < 0: #flip - make the projection positive (thinking your'e inside a RC)
                    proj = -proj*np.abs(alpha) if s < 0 else proj*np.abs(alpha)
                elif alpha_sign > 0: #enhance - make the projectio negative (thinking your'e outside of RC)
                    proj = proj*np.abs(alpha) if s < 0 else -proj*np.abs(alpha)
                
                states_projected[mask_idx_bert] += proj               
        ### SELECTIVE-FLIPPING-END 
        
     
        
        #norm_after = torch.norm(states_projected, dim = 1, keepdim = True)
        #NORMALIZE = False
        #if NORMALIZE:
        #    states_projected = (states_projected/norm_after)*norm_original
             
        next_layers = forward_from_specific_layer(self.model, layer, states_projected.unsqueeze(0)).squeeze(1)
        last_after_batchnorm = next_layers[-1]
        vec = last_after_batchnorm[mask_idx_bert]

        top_k = 500
        logits = np.dot(self.out_embed, vec)
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)
        top_words = self.tokenizer.convert_ids_to_tokens(top_k_indices)[:]
        predictions_dict = defaultdict(dict)

        for i, pred_idx in enumerate(top_k_indices):
            predicted_token = self.tokenizer.convert_ids_to_tokens([pred_idx])[0]
            #print(predicted_token)
            token_weight = top_k_weights[i]
            predictions_dict[predicted_token]["prob"] = token_weight.detach().cpu().numpy().item()
            predictions_dict[predicted_token]["rank"] = i

        w = sentence.split(" ")[mask_index]
        #print(w, predictions_dict[w], "applying projection?", P2 is not None)
        #print("----------------------------")
        top_k_weights = ['%.7f' % x for x in top_k_weights.detach().cpu().numpy()]
        return predictions_dict, vec, top_words, top_k_weights, orig2tok, tokenized_text


def collect_bert_states(bert, data: List[Tuple], layer=-1, P=None, P2=None, alpha=None,ws=None,project=False):
    data_with_states = []

    for i, d in enumerate(data):

        sent = d["sent"]# + " ."
        word_position = d["verb_index"]
        correct, wrong = d["correct_verb"], d["wrong_verb"]
        
        predictions_dict, vec, top_words, top_probs, orig2tok, bert_tokens = bert.encode(sent, mask_index=word_position,
                                                                              layer=layer, P=P, P2=P2, alpha = alpha,ws=ws,project=project)

        dict_data = {} #d.copy()

        dict_data["top_preds"] = list(zip(top_words[:10], top_probs[:10]))
        #print(dict_data["top_preds"][:250])
        #print("======================================")

        
        dict_data["sentence"] = sent
        dict_data["correct_word"] = correct
        dict_data["wrong_word"] = wrong
        dict_data["verb_index"] = word_position

        if correct in predictions_dict:
            dict_data["correct_word_prob"] = predictions_dict[correct]["prob"]
            dict_data["correct_word_rank"] = predictions_dict[correct]["rank"]
        else:
            dict_data["correct_word_prob"] = None
            dict_data["correct_word_rank"] = None
            
        if wrong in predictions_dict:
            dict_data["wrong_word_prob"] = predictions_dict[wrong]["prob"]
            dict_data["wrong_word_rank"] = predictions_dict[wrong]["rank"]
        else:
            dict_data["wrong_word_prob"] = None
            dict_data["wrong_word_rank"] = None
        
        if dict_data["correct_word_rank"] is not None and dict_data["wrong_word_rank"] is not None:
        
            dict_data["success"] = dict_data["correct_word_rank"] < dict_data["wrong_word_rank"]
        
        elif dict_data["correct_word_rank"] is not None and (dict_data["wrong_word_rank"] is None):
        
            dict_data["success"] = True
        
        else:
            dict_data["success"] = False

        #print(dict_data)
        #print("------------------")
        data_with_states.append(dict_data)

    return data_with_states
