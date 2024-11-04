import numpy as np
import sentencepiece as spm

PAD, UNK, BOS, EOS = '<pad>', '<unk>', '<bos>', '<eos>'
LS, RS, SP = '<s>', '</s>', ' '
BINST, EINST = '<INST>', '</INST>'
BSYS, ESYS = '<SYS>', '</SYS>'


class Tokenizer:
    def __init__(self, filename, min_occur_cnt, specials=None):
        idx2token = [PAD, UNK, BOS, EOS] + \
            [LS, RS, SP, BINST, EINST, BSYS, ESYS] + \
            (specials if specials is not None else [])
        for line in open(filename, encoding='utf-8').readlines():
            try:
                token, cnt = line.strip().split()
            except:
                continue
            if int(cnt) >= min_occur_cnt:
                idx2token.append(token)
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def random_token(self):
        return self.idx2token(1 + np.random.randint(self.size - 1))

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)

    def encode(self, x):
        return self.token2idx([w for w in x])

    def decode(self, x):
        return ''.join(self.idx2token(x))


if __name__ == "__main__":
    tokenizer = Tokenizer(model_path='./model/m.model')

    sample_text = "南京航空航天大学是南京的一所双一流高校。Nanjing University of Aeronautics and Astronautics is a double first-class university in Nanjing."
    tokens = tokenizer.tokenize(sample_text)
    print("Tokens:", tokens)

    detokenized_text = tokenizer.detokenize(tokens)
    print("Detokenized Text:", detokenized_text)

    print(tokenizer.tokenize(sample_text))
