import json
from multiprocessing import Pool
from collections import Counter
import sentencepiece as spm
from tqdm import tqdm
BUFSIZE = 10000

ttype = "char"


def process(doc):
    res = [w for w in doc]
    return res


def save(cnt, docs, nprocessors):
    res = pool.map(process, docs, len(docs) // nprocessors)
    all_lines = []
    for xs in res:
        all_lines.extend(xs)
    for x in all_lines:
        cnt.update(x)


if ttype == "char":
    cnt = Counter()
    nprocessors = 20
    pool = Pool(nprocessors)
    docs = []
    with open("./data/train.txt", "r") as f:
        total_lines = sum(1 for line in f)  # 计算总行数
        f.seek(0)  # 回到文件开头
        for line in tqdm(f, total=total_lines, desc="Processing lines"):
            line = line.strip()
            if not line:
                continue
            line = json.loads(line)['text']
            if not line:
                continue
            docs.append(line)

            if len(docs) == BUFSIZE:
                save(cnt, docs, nprocessors)
                docs = []
                # print(BUFSIZE)

        if len(docs) > 0:
            save(cnt, docs, nprocessors)
            print(len(docs))
    print("vocab")
    with open("./model/vocab.txt", 'w', encoding='utf8') as f:
        for x, y in cnt.most_common():
            f.write(x + '\t' + str(y) + '\n')
    print("done")

elif ttype == "bpe":
    spm.SentencePieceTrainer.train(input='./data/train.txt', model_prefix='m', vocab_size=32000,
                                   character_coverage=1.0, model_type='bpe',
                                   user_defined_symbols=['<pad>', '<bos>', '<eos>', '<mask>', '<INST>', '</INST>', '<SYS>', '</SYS>'])
else:
    pass
