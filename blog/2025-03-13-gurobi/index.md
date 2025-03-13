---
slug: gurobi
title: Gurobi 101
authors: eason
tags: [or]
draft: false
---

本文介绍了如何熟悉Gurobi，并展开学习。

<!-- truncate -->

## 安装

### 通过Docker使用Gurobi

我们可以运行如下命令启动基于Docker的Python Jupyter的是运行环境。
该环境默认提供了多种的Python Notebook案例，我们可以择取来学习Gurobi。

```bash
docker run -p 10888:8888 gurobi/modeling-examples
```

运行命令后，我们打开：
[http://localhost:10888/](http://localhost:10888/)来查看Jupyter环境。
如下图所示：

![Jupyter landing page](./jupyter-landing-page.png)

我们打开任意一个文件夹，即可开始进行Gurobi的案例研究。