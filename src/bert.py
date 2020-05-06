import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, RobertaModel, RobertaForMaskedLM, RobertaTokenizer, RobertaConfig
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AlbertTokenizer, AlbertModel, AlbertConfig
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig
import numpy as np
from typing import List, Tuple, Dict
import tqdm

class BertEncoder(object):

    def __init__(self, device='cpu', model="bert"):

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

    def encode(self, sentence: str, layers: List[int], final_transform: bool = True, pos_ind=-1, mask_prob=1.0):

        tokenized_text, orig2tok = self.tokenize(sentence.split(" "))
        pos_ind_bert = orig2tok[pos_ind]
        if np.random.random() < mask_prob:
            tokenized_text[pos_ind_bert] = "[MASK]"
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            predictions = torch.cat([outputs[1][layer][0] for layer in layers], axis=-1)  # .detach().cpu().numpy()
            if final_transform:
                predictions = self.final_transform(predictions)
            predictions = predictions.detach().cpu().numpy()

            """
            if layer >= 0:
                predictions = outputs[2][layer].detach().cpu().numpy()
            else:
                concat = torch.cat(outputs[2], axis = 0)
                concat = concat[:7, :, :]
                predictions = concat.reshape(concat.shape[1], concat.shape[0] * concat.shape[2])

                print(predictions.shape)
                print("----------------------------")
                #predictions = torch.sum(concat, axis = 0).detach().cpu().numpy()
            """
            return (predictions.squeeze(), orig2tok, tokenized_text)


