---
slug: manim
title: 使用Manim制作科普视频
authors: eason
tags: []
draft: false
---

本文介绍了如何使用Manim数学动画引擎来制作算法可视化视频，以Dijkstra最短路径算法为例。

<!-- truncate -->

## 使用Docker启动Manim镜像

```shell
docker run -v ~/manim:/manim  -it -p 8888:8888 manimcommunity/manim jupyter lab --ip=0.0.0.0 --ServerApp.token='manim'
```

这个命令会下载并启动Manim的Docker镜像，将本地的~/manim目录挂载到容器中，并在8888端口启动JupyterLab，可以通过http://localhost:8888访问，密码是'manim'。

## Dijkstra最短路径算法可视化

Dijkstra算法是一种经典的图论算法，用于找出加权图中从一个节点到所有其他节点的最短路径。下面我们将使用Manim创建一个动画，演示这个算法的工作过程。

### 算法原理

Dijkstra算法的基本思想是：

1. 初始化：将起点距离设为0，其他所有点距离设为无穷大
2. 从未访问的节点中选择距离最小的节点
3. 更新该节点的所有邻居的距离
4. 重复步骤2和3，直到所有节点都被访问

### 实现代码

我们创建了一个完整的Manim动画来展示Dijkstra算法的过程。代码存放在`dijkstra.py`文件中：

```python
# 代码片段预览
class DijkstraAlgorithm(Scene):
    def construct(self):
        # 创建图形
        self.create_graph()
        # 运行算法
        self.run_dijkstra()
```

完整代码详见仓库中的`dijkstra.py`文件。

### 运行动画

在JupyterLab中，可以通过以下步骤运行动画：

1. 打开`dijkstra_demo.ipynb`笔记本
2. 运行笔记本中的代码单元格
3. 观看生成的动画

或者直接在命令行中运行：

```shell
# 低质量（快速渲染，适合测试）
manim -ql dijkstra.py DijkstraAlgorithm

# 高质量（最终渲染）
manim -qh dijkstra.py DijkstraAlgorithm
```

### 动画效果

动画将展示以下内容：

- 创建一个带权重的有向图
- 从节点A开始运行Dijkstra算法
- 展示算法如何逐步探索图中的节点
- 高亮显示当前正在考虑的边和节点
- 更新每个节点的距离标签
- 最终显示从起点到所有其他节点的最短路径

## 自定义动画

可以通过修改代码来自定义动画：

1. 更改图的结构（节点位置和边的权重）
2. 更改起始节点
3. 调整动画的时间和效果
4. 更改视觉样式（颜色、大小等）

Manim是一个功能强大的工具，可以创建各种数学和算法的可视化。通过这个Dijkstra算法的例子，我们展示了如何使用它来创建清晰、直观的教学动画。 