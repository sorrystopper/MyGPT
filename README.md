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
1. 以上代码都能在我服务器上运行，且看起来效果还不错，如果有问题，可以Q我 or Email `1377765332@qq.com`。

# 更新！！！
1. 本次实验更新了`DPO`和`Application`；
2. `DPO`分别对自己的`sft`模型和`Qwen`系列模型进行了`DPO`更新；
3. `Application`选择的是文本分类，直接使用`Qwen`系列的模型，分别实现使用`zero-shot`的`prompt`进行对话生成的文本分类，和直接使用`Qwen` `Model`代码中的`AutoModelForSequenceClassification`进行直接分类训练。

## DPO
1. 分别在`DPO_MyGPT`和`DPO_Qwen`文件夹中(训练数据自己准备)
```
cd DPO_MyGPT 或者 cd DPO_Qwen
训练：
python train.py
测试
python inference.py
```
2. 自己的`sft`模型`dpo`后好像训崩了(应该是代码写的依托，但是懒得改了)，但是`Qwen`模型`dpo`后效果very good；

## Application
1. 在`Text_classification`文件夹中，因为我对数据集也做了一些处理，所以直接传上去了，在`toutiao_cat_data`文件夹中；
2. zero-shot的对话生成进行分类：
```
cd Text_classification
训练：
python train.py
测试
python inference.py
```
3. 直接使用`Qwen`定制的`AutoModelForSequenceClassification`进行文本分类：
```
cd Text_classification
训练：
python train_v2.py
测试
python inference_v2.py
```