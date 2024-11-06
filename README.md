# 注意！！！
1. 我是用V100跑了9h pretrain后的模型+一坤时sft后的模型（人工看起来效果还不错？），因为是在服务器上跑的，懒得下载到本地（下起来可能会导致vscode卡住），要注意的是`pretrain`中的`bash step_02.sh`中的`CUDA_VISIBLE_DEVICE`、`world_size`和`gpus`参数都需要根据自己的卡来调整
2. 如果你要参考我的代码，可以提前和我说一声？
3. pretrain和sft的模型参数我没有放到这里，需要的话可以Q我。
4. 有问题可以Q我 or Email at `1377765332@.com`。

## train_tokenizer
train_tokenizer代码在`train_tokenizer.py`中，用`bash step_01.sh`运行，这里使用的是`char-based`的方法，后面看时间情况改成使用`bpe`的方法（看来是没时间弄了）。

## pretrain
pretrain代码在`pretrain.py`中，用`bash step_02.sh`运行。
注意：
1. `--train_type pretrain`来指定模型进行预训练。


## Evaluation
用`python pretrain.py`运行。
注意：
1. 在`pretrain.py`代码中自己修改想要评测模型的位置。
2. 修改`eval_type`变量来选择是对`pretrain`测试还是对`sft`测试

## sft
用`bash step_02.sh`运行。
注意：
1. `--train_type sft`来指定模型进行`supervised finetune`。

## ceval
执行
```
cd ceval
bash ceval.sh
```
注意：
1. 在`ceval.sh`中修改自己的模型路径和参数。
2. 修改了`generate.py`，没懂为什么老师是用`logits=logits[0][0]`，不应该是根据最近生成的token获得的词典大小的logits来预测ABCD吗？

## 总结
1. 以上代码都能在我服务器上运行，且看起来效果还不错，如果有问题，可以Q我 or Email `1377765332@.com`。