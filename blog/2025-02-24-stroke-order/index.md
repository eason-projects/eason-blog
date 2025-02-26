---
slug: stroke-order
title: 强化学习中文笔画顺序预测
authors: eason
tags: [ml, rl]
draft: true
---

我们看到随着具身智能（机器人）的蓬勃发展，
越来越多的机械臂应用面世。

在这些里面，就有通过驱动机器人来书写汉字的应用。

为了探索此类的应用，我们简化任务，我们希望通过输入图片来预测图片中汉字的笔画顺序。
以作为未来通过机器人书写汉字的基础。

<!-- truncate -->

## 提示词

```
我们希望构建一个RL的算法，通过输入汉字的图片，算法需要正确的输出汉字的笔画（笔顺）。

如何构建？选用什么样的方式实现并训练呢？

注：
1. 我们只要求预测笔画和顺序，不要求预测笔画的起始和结束位置。（不需要预测笔画的区域或者位置）
2. 推荐使用Stable-Baselines3。
3. 详细解释下状态（state）的设计。
4. 在训练的过程中，我们可以知道每一个汉字的笔画顺序。但是在使用模型结果的时候，我们不知道画面中的汉字的正确笔画，我们希望完全依赖模型给我们结果。
```

## Setup

```bash
# Cretae a venv
python3 -m venv venv

source venv/bin/activate

pip3 install -r requirements.txt
```

## AI提供的建议

1. https://www.wenxiaobai.com/share/chat/f16629de-6eab-4efd-8b4f-b368f7a77e36
2. http://www.moe.gov.cn/jyb_sjzl/ziliao/A19/202103/W020210318300204215237.pdf
3. 