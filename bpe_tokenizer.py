import sentencepiece as spm
import numpy as np
PAD, UNK, BOS, EOS = '<pad>', '<unk>', '<bos>', '<eos>'


class BPE_Tokenizer:
    def __init__(self, model_path, specials=None):
        # 加载BPE模型
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        # 定义特殊符号
        self.specials = [PAD, UNK, BOS, EOS] + \
            (specials if specials is not None else [])

        # 创建 token 到 idx 的映射
        self._token2idx = {token: idx for idx,
                           token in enumerate(self.specials)}
        self._idx2token = self.specials

        # 通过BPE模型的词汇表来填充映射
        vocab_size = self.sp.get_piece_size()
        for i in range(vocab_size):
            token = self.sp.id_to_piece(i)
            self._token2idx[token] = len(self._token2idx)
            self._idx2token.append(token)

        # 设置特殊符号的idx
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]
        self._eos_idx = self._token2idx[EOS]

    @property
    def size(self):
        return self.sp.get_piece_size()

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def random_token(self):
        # 随机选择一个token
        return self.idx2token(np.random.randint(self.size - 1) + 1)

    def idx2token(self, x):
        return self.sp.id_to_piece(x)

    def token2idx(self, x):
        return self.sp.piece_to_id(x)

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, indices):
        return self.sp.decode(indices)


if __name__ == "__main__":
    tokenizer = BPE_Tokenizer(model_path="./model/m.model")

    sample_text = "南京航空航天大学是南京的一所双一流高校。"
    tokens = tokenizer.encode(sample_text)
    print("Tokens:", tokens)

    tokens = tokenizer.sp.encode(sample_text, out_type=str)
    print("Str Tokens:", tokens)

    detokenized_text = tokenizer.decode(tokens)
    print("Detokenized Text:", detokenized_text)
