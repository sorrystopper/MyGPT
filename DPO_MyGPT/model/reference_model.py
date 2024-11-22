from transformers import AutoModelForCausalLM
import torch.nn as nn
import torch
import sys
import os

sys.path.append('/home/xli/skw/MyGPT')
from mygpt import myGPT
from tokenizer import Tokenizer
from generator import generate_dpo


class ReferenceModel(nn.Module):
    def __init__(self, config):
        super(ReferenceModel, self).__init__()
        # self.config = config
        # self.reference_model = AutoModelForCausalLM.from_pretrained(self.config.gpt_model, torch_dtype="auto").to(
        #     self.config.device)
        self.model_path = config.gpt_model
        self.vocab_path = config.vocab_path
        ckpt = torch.load(self.model_path, map_location='cpu')
        lm_args = ckpt['args']
        self.device = config.device
        self.tokenizer = Tokenizer(self.vocab_path, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
        self.model = myGPT(config.device, self.tokenizer, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads,
                           lm_args.dropout, lm_args.layers)
        self.model.load_state_dict(ckpt['model'])
        self.model = self.model.to(config.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        inputs, response, logits = generate_dpo(self.model, self.tokenizer, self.device, 40, 250, \
                                                input_ids, attention_mask)
        return logits


