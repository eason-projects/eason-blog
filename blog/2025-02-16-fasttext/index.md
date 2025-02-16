---
slug: fasttext
title: 使用MLflow和Ray训练fastText
authors: eason
tags: [ml, mlflow, nlp]
---

fastText是Facebook研发的一款针对NLP领域的解决方案。

其主要提供了文本分类和词向量学习两大功能。
其核心思想是将整句话的词向量叠加平均作为文本表示，
并使用softmax分类器进行分类。

我们通过本文介绍一下如何使用MLflow以及Ray来训练我们的fastText模型。

<!-- truncate -->

## fastText介绍

fastText作为一个高效的文本分类和词向量表示工具，fastText具有以下特点：

1. **训练速度快**：能够在普通多核CPU上几秒内处理数十亿个词，训练数百万个文本分类器
2. **效果优异**：在文本分类任务中取得与深度学习模型相当的精度
3. **资源占用少**：相比深度学习模型，fastText对硬件要求低，且模型文件小
4. **多语言支持**：支持294种语言的词向量训练

## 训练目标

在本文中，我们会针对[淘宝客服对话数据](https://github.com/cooelf/DeepUtteranceAggregation/)
这个数据集进行处理，我们希望可以训练一个分类器，
来对任意对话文本区分是客户还是客服的消息。

## 数据准备

### 数据文件格式

我们通过上面的网址下载后，可以看到有3个数据文件，分别是 `train.txt`、 `dev.txt`以及`test.txt`。

打开任意的文件，其内部数据如下：

```plaintext
1	在 吗	您好	现在 拍 几天 能 到 辽宁	这个 不 一定 哦	大概 几天 不 知道 么	一般 情况 下 3 到 5 天 左右
0	在 吗	您好	现在 拍 几天 能 到 辽宁	这个 不 一定 哦	大概 几天 不 知道 么	亲 不会 的 呢 您 放心
```

每一行表示为：

- `1`或`0`：正确的对话流程以及错误的对话流程。
- 循环（用`\t`来隔开）：
  - 客户问题
  - 客服回答

比如上面的数据样本的第一行：

```
1	--> 正确的对话
在 吗	--> 客户问题
您好	--> 客服回答
现在 拍 几天 能 到 辽宁	这个 不 一定 哦	大概 几天 不 知道 么	--> 客户问题
一般 情况 下 3 到 5 天 左右	--> 客服回答
```

### fastText要求的数据格式

fastText有自己独立的数据格式，其输入为文本文件，其每一行的数据格式为：

```
分类1 分类2 分类... 文本行
```

即，每一行可以关联多个分类，然后分类信息以及文本行信息以空格隔开。

其分类表示有独特的要求，比如我们希望构建两个分类：

- `seller`：客服
- `customer`：客户

那么其fastText表示为： `__label__customer`和`__label__seller`。

因此，我们需要将上述的原始数据文件，每行进行解析，并按照存储如下的格式的文件，如：

```
__label__customer 您好咱们这边如果能提升您的销量利润您会考虑跟我们合作吗
__label__customer 亲你家什么时候还有活动啊
__label__customer 我的订单怎么两天了没什么变化啊在么
__label__customer 好麻烦你改下谢谢
__label__customer 明天可以发货
__label__seller 有的哦满68送95g猪肉脯一袋
__label__customer 这个买两个有优惠吗
__label__customer 亲亲已下单买这么多请掌柜的多送点小礼物啊
__label__customer 嗯嗯了解了哦
__label__customer 我买10个付款的时候怎么不打折呢
__label__seller 您这边提交订单看看哦系统自动改价的和芒果干一样的
__label__seller 不好意思亲可能快递途中挤压造成的这边退亲2元差价亲看可以吗
__label__customer 亲有原味瓜子么
```

### 处理代码

我们可以构建下面的两个函数，来读取原始的数据文件，然后逐行按照上面的格式，构建一个包含了客户和客服的数据文件。
并按照fastText的格式，保存成训练或者验证文件。

```python
def load_data(path):
    with open(path, 'r') as f:
        data = f.readlines()

    customer_utterances = []
    seller_utterances = []

    for idx, line in tqdm(enumerate(data)):
        utterances = line.strip().split('\t')
        utterances = [re.sub(r'[\s\n\t]', '', utterance) for utterance in utterances]

        if utterances[0] == '1':
            customer_utterances += utterances[1::2]
            seller_utterances += utterances[2::2]

    # Remove utterances only contain digits. If it contains digits and other characters, keep it.
    customer_utterances = [utterance for utterance in customer_utterances if not re.match(r'^[0-9]+$', utterance)]
    seller_utterances = [utterance for utterance in seller_utterances if not re.match(r'^[0-9]+$', utterance)]

    # Only keep utterances with more than 5 characters
    customer_utterances = [utterance for utterance in customer_utterances if len(utterance) > 5]
    seller_utterances = [utterance for utterance in seller_utterances if len(utterance) > 5]

    # If the utterance are the same in both customer and seller, remove duplicates using sets
    customer_set = set(customer_utterances)
    seller_set = set(seller_utterances)
    
    # Remove duplicates that appear in both sets
    unique_customer = list(customer_set - seller_set)
    unique_seller = list(seller_set - customer_set)
    
    return unique_customer, unique_seller

def generate_fasttext_data(customer_utterances, seller_utterances, output_path):
    """Generate FastText training data with labels."""
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write customer utterances
        for utterance in customer_utterances:
            f.write(f"__label__customer {utterance}\n")
        
        # Write seller utterances
        for utterance in seller_utterances:
            f.write(f"__label__seller {utterance}\n")
    
    print(f"FastText training data saved to {output_path}")
    print(f"Total samples: {len(customer_utterances) + len(seller_utterances)}")
    print(f"Customer samples: {len(customer_utterances)}")
    print(f"Seller samples: {len(seller_utterances)}")
```

文件处理后，共有训练数据238,275条，其中客户对话138,429条，客服对话99,846条。

