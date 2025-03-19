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

我们可以运行如下命令启动基于Docker的Python JupyterLab的运行环境。
该环境默认提供了多种的Python Notebook案例，我们可以择取来学习Gurobi。

```bash
docker run -p 10888:8888 gurobi/modeling-examples
```

运行命令后，我们打开：
[http://localhost:10888/lab](http://localhost:10888/lab)来查看JupyterLab环境。
如下图所示：

![Jupyter landing page](./jupyter-landing-page.png)

我们打开任意一个文件夹，即可开始进行Gurobi的案例研究。


## 二进制整数规划

我们首先来尝试解决一个非常经典的[背包问题](https://zh.wikipedia.org/wiki/%E8%83%8C%E5%8C%85%E9%97%AE%E9%A2%98)。

假设，我们要去旅行，要携带一些物品，而这些物品有重量和对于我们的价值。
那么我们如果用一个背包来携带这些物品的话，由于背包的容量限制，
导致我们必须最大化的选择一些物品来携带。

假设我们有如下的物品：

| 物品名称 | 重量 | 价值 |
| -------- | ---- | ---- |
| 手电筒   | 1    | 5    |
| 睡袋     | 3    | 12   |
| 食物     | 2    | 8    |
| 水       | 4    | 15   |

那么我们的思路是，通过Gurobi来定义变量（Variables）、限制条件（Constraints）以及优化目标（Objective），
来找到针对上面的问题最优的答案。

首先，我们来创建一个Gurobi Python模型：

```python
# Import Gurobi Python 
import gurobipy as gp

model = gp.Model("Knapsack")
```

然后我们将上述问题的一些参数进行定义，比如物品的名称、重量和价值等：

```python
# Set items with weight and value
items = {
    'flashlight': {'weight': 1, 'value': 5},
    'sleeping_bag': {'weight': 3, 'value': 12},
    'food': {'weight': 2, 'value': 8},
    'water': {'weight': 4, 'value': 15}
}

# Set the max capacity
CAPACITY = 7
```

接着，我们给Gurobi模型添加相关的变量，我们通过二进制的表示（1是携带该物品，0是不携带该物品）来表示最后的携带状态。

```python
# Add variables
x = model.addVars(items.keys(), vtype=gp.GRB.BINARY, name="select")
```

现在我们需要添加上述问题中的限制条件，比如所选择的物品不能超过我们背包可容纳的上限。

```python
model.addConstr(
    gp.quicksum(items[i]['weight'] * x[i] for i in items) <= CAPACITY,
    name="capacity"
)
```

最后，我们需要添加我们所期望的优化目标，也就是我们的Objective。

```python
model.setObjective(
    gp.quicksum(items[i]['value'] * x[i] for i in items),
    gp.GRB.MAXIMIZE,
)
```

上述内容添加后，我们就可以开始优化求解的流程了。执行：

```python
model.optimize()
```

会触发优化过程，其输出结果如下所示：

```plaintext
Gurobi Optimizer version 12.0.1 build v12.0.1rc0 (linux64 - "Debian GNU/Linux 11 (bullseye)")

CPU model: Intel(R) Core(TM) i7-7820HQ CPU @ 2.90GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 8 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 1 rows, 4 columns and 4 nonzeros
Model fingerprint: 0xc569c671
Variable types: 0 continuous, 4 integer (4 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+00, 2e+01]
  Bounds range     [1e+00, 1e+00]
  RHS range        [7e+00, 7e+00]
Found heuristic solution: objective 25.0000000
Presolve removed 1 rows and 4 columns
Presolve time: 0.00s
Presolve: All rows and columns removed

Explored 0 nodes (0 simplex iterations) in 0.05 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 2: 28 25 

Optimal solution found (tolerance 1.00e-04)
Best objective 2.800000000000e+01, best bound 2.800000000000e+01, gap 0.0000%
```

我们可以通过如下的检查语句来查看模型最后结果的细节内容：

```python
# Get model's status, gp.GRB.OPTIMAL == 2
model.status

# Get the objective value
model.objVal

# Print each variables' value
for i in model.getVars():
    print(f'{i.varName}, {i.x}')
```

运行后，我们可以看到我们的模型最优结果是28，是如下的物品组合：

```plaintext
select[flashlight], 1.0
select[sleeping_bag], 0.0
select[food], 1.0
select[water], 1.0
```

也就是，需要携带手电筒、食物和水。不携带睡袋。

至此，我们的求解工作完成。
我们实现了一个非常简单的运筹优化求解的动作。

## 整数规划

在[社交媒体上](https://x.com/rainmaker1973/status/1901639842150125637)看到这样一个问题：

![Area of blue rectangle](./blue-rectangle.png)

根据已知条件求解蓝色区域的面积。

很显然，我们可以通过Gurobi来求解，具体代码如下：

```python
# Import Gurobi
import gurobipy as gp

# Create a new model
m = gp.Model("Size")

# Define variables
height = m.addVar(vtype=gp.GRB.INTEGER, name="height")
width = m.addVar(vtype=gp.GRB.INTEGER, name="width")
x1 = m.addVar(vtype=gp.GRB.INTEGER, name="x1")
x2 = m.addVar(vtype=gp.GRB.INTEGER, name="x2")

# Define constraints
m.addConstr(x1 + width == 7, "7m")
m.addConstr(x2 + width == 8, "8m")
m.addConstr(x1 * height == 20, "20m2")
m.addConstr(x2 * height == 25, "25m2")

# Define objective
m.setObjective(height * width, gp.GRB.MAXIMIZE)

# Solve
m.optimize()

# Print values
m.getVars()
```

我们模型输出的结果是：

```plaintext
Gurobi Optimizer version 12.0.1 build v12.0.1rc0 (linux64 - "Debian GNU/Linux 11 (bullseye)")

CPU model: Intel(R) Core(TM) i7-7820HQ CPU @ 2.90GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 8 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 2 rows, 4 columns and 4 nonzeros
Model fingerprint: 0x9c0c59c0
Model has 1 quadratic objective term
Model has 2 quadratic constraints
Variable types: 0 continuous, 4 integer (0 binary)
Coefficient statistics:
  Matrix range     [1e+00, 1e+00]
  QMatrix range    [1e+00, 1e+00]
  Objective range  [0e+00, 0e+00]
  QObjective range [2e+00, 2e+00]
  Bounds range     [0e+00, 0e+00]
  RHS range        [7e+00, 8e+00]
  QRHS range       [2e+01, 2e+01]
Presolve time: 0.00s
Presolved: 12 rows, 6 columns, 25 nonzeros
Presolved model has 3 bilinear constraint(s)

Solving non-convex MIQCP

Variable types: 2 continuous, 4 integer (0 binary)

Root relaxation: objective 1.500000e+01, 0 iterations, 0.00 seconds (0.00 work units)

    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

*    0     0               0      15.0000000   15.00000  0.00%     -    0s

Explored 1 nodes (0 simplex iterations) in 0.09 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 1: 15 

Optimal solution found (tolerance 1.00e-04)
Best objective 1.500000000000e+01, best bound 1.500000000000e+01, gap 0.0000%
```

同时变量数值为：

```plaintext
[<gurobi.Var height (value 5.0)>,
 <gurobi.Var width (value 3.0)>,
 <gurobi.Var x1 (value 4.0)>,
 <gurobi.Var x2 (value 5.0)>]
```

因此，蓝色区域的面积为 **3x5 = 15**平方米。


## 分支定界算法

首先对问题进行定义，然后通过线性规划松弛（Linear Programming Relaxation）来获得小数解（作为初始的上界或者下界）。
松弛的含义就是暂时去掉结果必须是整数的限制，先通过小数来求一个在当前可行域内的全局最优小数解，作为后续分支的参考界限。

然后，通过分支将问题拆解为多个子问题，针对某个变量的取值添加整数约束，从而缩小解空间，探索可能的整数解。
依次遍历部分子问题，并通过剪枝（Pruning），提前终止那些目标函数值比当前已知界更差的分支，以减少计算量。

通过不断更新上界和下界，逐步收敛，直到找到全局最优的整数解，或者所有分支都被探索或剪枝，算法终止。

![Branch and Bound Algorithm Example](./branch-and-bound-youtube-screenshot.png)
*[Branch and bound algorithm example](https://youtu.be/cEcS13Ku1i8?t=516)*



## 参考资料

- Gurobi官方入门指南: [https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer)
- Branch and bound algorithm example: [https://www.youtube.com/watch?v=cEcS13Ku1i8](https://www.youtube.com/watch?v=cEcS13Ku1i8)
