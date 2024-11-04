# train_tokenizer
train_tokenizer代码在`train_tokenizer.py`中，用`bash step_01.sh`运行，这里使用的是`char-based`的方法，后面看时间情况改成使用`bpe`的方法。

# pretrain
pretrain代码在`pretrain.py`中，用`bash step_02.sh`运行。
注意：
1. `--train_type pretrain`来指定模型进行预训练。


# Evaluation
用`python pretrain.py`运行。
注意：
1. 在`pretrain.py`代码中自己修改想要评测模型的位置。

# sft
用`bash step_02.sh`运行。
注意：
1. `--train_type sft`来指定模型进行`supervised finetune`。

# ceval
执行
```
cd ceval
bash ceval.sh
```
注意：
1. 在`ceval.sh`中修改自己的模型路径和参数