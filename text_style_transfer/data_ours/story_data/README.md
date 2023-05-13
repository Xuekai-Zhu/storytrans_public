# 基于故事大纲的条件生成数据集

### 简介

Outline-conditioned Generation (OutGen) 数据集由清华大学CoAI小组收集。



### 数据规模

训练集：1,456，验证集：242，测试集：729



### 数据样例

```
{
	"story":
		"有个人把神像放在驴子背上，赶着进城。凡是遇见他们的人都对着神像顶礼膜拜。驴子以为人们是向它致敬，便洋洋得意，大喊大叫，再也不肯往前走了。结果挨了驴夫狠狠的一棍。", 
	"outline":
  	["对着神像顶礼膜拜", "再也不肯往前走", "神像放在驴子", "赶着进城", "驴夫狠狠", "洋洋得意", "大喊大叫", "遇见"], 
	"title":
		"运神像的驴子"
}
```

- "title" (`str`)：输入的故事标题。
- "outline"（`list of str`）：输入的故事大纲（一系列无序的短语）。
- "story" (`str`)：期待输出的故事。



### 评测代码使用

```shell
python eval.py prediction_file test_private_file
```

- 预测结果需要和评测代码保持一样的格式
- 依赖：rouge\=\=1.0.0，jieba=0.42.1, nltk\=\=3.6.2, numpy\=\=1.20.3
- 评测指标为bleu-1, bleu-2, bleu-3, bleu-4, distinct-1, distinct-2, distinct-3, distinct-4, repetition-1, repetition-2, repetition-3, repetition-4，输出结果为字典格式：

```python
{'bleu-1': '_', 'bleu-2': '_', 'bleu-3': '_', 'bleu-4': '_', 'repetition-1': '_', 'distinct-1': '_', 'repetition-2': '_', 'distinct-2': '_', 'repetition-3': '_', 'distinct-3': '_', 'repetition-4': '_', 'distinct-4': '_'}
```

