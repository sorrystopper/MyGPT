import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from peft import LoraConfig, PeftModel
from config import LoraArguments
import sys
import os
sys.path.append('/home/xli/skw/MyGPT')
from mygpt import myGPT
from tokenizer import Tokenizer
from generator import generate_dpo


class LoraModel(PeftModel):
    def __init__(self, config: Config, model):
        lora_args = LoraArguments()
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            task_type="CAUSAL_LM",
        )
        super().__init__(model, lora_config)
        if lora_args.is_reload_trained_params:
            super().from_pretrained(model, config.save_lora_path)
        for name, module in self.named_modules():
            if 'lora_' in name:
                for param in module.parameters():
                    param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        res = super().forward(input_ids, attention_mask, output_hidden_states=True)
        return res.logits


class Model(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # model = AutoModelForCausalLM.from_pretrained(config.gpt_model).to(config.device).eval()
        # self.model = LoraModel(config, model)
        # self.tokenizer = AutoTokenizer.from_pretrained(config.gpt_model)
        self.model_path = config.gpt_model
        self.vocab_path = config.vocab_path
        ckpt = torch.load(self.model_path, map_location='cpu')
        lm_args = ckpt['args']
        self.device = config.device
        self.tokenizer = Tokenizer(self.vocab_path, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
        self.model = myGPT(config.device, self.tokenizer, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads,lm_args.dropout, lm_args.layers)
        self.model.load_state_dict(ckpt['model'])
        self.model = self.model.to(config.device)
        self.model.train()

    def forward(self, input_ids, attention_mask):
        # logits = self.model(input_ids, attention_mask)
        inputs, response, logits = generate_dpo(self.model, self.tokenizer, self.device, 40, 250,\
                                                input_ids,attention_mask)
        return logits

