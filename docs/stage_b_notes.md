# Stage B — Fashion-MNIST 图片分类完整流程

## 1. 这一阶段学会了什么

这一阶段用 PyTorch 独立完成一个 MLP 图片分类程序：加载数据、划分训练与验证集、组织 batch、定义模型、训练、统计指标、保存最佳参数、恢复模型，以及评估测试集。先在 Notebook 中逐步实现，再整理为 `scripts/stage_b_fashion_mnist.py`。

目标是理解并写通整个过程，不是追求 Fashion-MNIST 的最高准确率。本阶段没有使用 CNN，也没有开展系统性的超参数搜索。

按我自己的理解，最初的流程是：

> 数据 → 数据载入 → 确定批次大小 batch_size → 取出一批数据 → 数据平摊，把每张图片摊开 → 模型预测 → 计算 loss → 计算梯度 → 更新参数。

完整流程还需要补上验证和测试：

每个 epoch 内先遍历训练集更新参数，再遍历验证集计算指标，根据验证准确率决定是否保存。所有 epoch 完成后，加载最佳 checkpoint，在测试集上评估。

## 2. Dataset、数据划分与 DataLoader

Fashion-MNIST 是服装图片数据集，不是手写数字 MNIST。每张图片是 28 × 28 的灰度图，共有 10 个类别。标签是类别编号，例如 9 表示 Ankle boot（短靴），不表示图片里写着数字 9。

本阶段最终的数据划分如下：

| 数据部分 | 数量 | 用途 |
|---|---:|---|
| Training set，训练集 | 54,000 | 计算梯度、更新参数 |
| Validation set，验证集 | 6,000 | 每轮评估、选择最佳模型 |
| Test set，测试集 | 10,000 | 对选定的模型做最终评估 |

官方训练部分有 60,000 张图片，使用 `random_split` 划分成 54,000 和 6,000；官方测试部分通过 `train=False` 加载。

```python
train_subset, val_subset = random_split(
    train_dataset,
    [54000, 6000],
    generator=torch.Generator().manual_seed(42),
)
```

这个 seed 固定的是数据划分，不会同时固定模型初始化和训练时的打乱顺序。之前用全部 60,000 张图片练习训练循环的模型不能直接用于新的验证实验，因为其中的验证图片已经参与过训练。正式划分后需要重新创建模型和 optimizer。

Dataset 提供单条样本，`random_split` 负责划分，DataLoader 负责组批和加载，它们是不同的职责。

### 一张图片与一个 batch

我当时问：

> 我数据集里的照片不应该是 [1,28,28] 吗，哪来的 32 批次，批次不是靠 DataLoader 干出来的吗？

单张图片确实是 `[1, 28, 28]`。前面的 32，是 DataLoader 把 32 张图片放到同一个 batch 后产生的。

| 对象 | Shape | 类型或含义 |
|---|---|---|
| 单张 image | `[1, 28, 28]` | float32，1 个灰度通道 |
| 一批 images | `[32, 1, 28, 28]` | 32 张图片 |
| 一批 labels | `[32]` | int64 类别编号 |

`ToTensor()` 将这里的图片转成 Tensor，像素值缩放到 `[0, 1]`。32 是每批的样本数，不是批次数量。

```python
for images, labels in train_set:
    ...
```

这里按位置解包：第一个变量接收图片，第二个接收标签。Python 不会根据变量名字决定内容。模型接收的是 `images`，不是整个 DataLoader。

需要批次编号时可以写：

```python
for batch_index, (images, labels) in enumerate(train_set):
    ...
```

`enumerate` 只是添加编号。for 循环不断从 DataLoader 的迭代器取下一批，变量名没有改变，但其中的数据已经换了。

### num_workers 与 shuffle

`num_workers` 指加载数据的工作进程数量，不是 GPU 数量，也不是 batch 数量。当前使用 `num_workers=0`，在主进程中加载数据。

训练 DataLoader 使用 `shuffle=True`，每轮重新打乱训练样本顺序；验证和测试使用 `shuffle=False`，便于固定评估顺序和查看预测图片。

## 3. Flatten 与模型结构

“组批”和“展平”不是同一件事。DataLoader 先组织图片，再由模型把每张图片摊开：

```python
flat_images = images.flatten(start_dim=1)
```

Shape 从 `[B, 1, 28, 28]` 变成 `[B, 784]`。保留第 0 维的 batch，从第 1 维开始合并后面的维度。不能对 DataLoader 调用 flatten，要对取出的 Tensor 调用。

Flatten 没有可训练参数，不改变像素值，也不会把原始数据集改掉。展平没有丢失像素值或固定的位置对应关系，但全连接层没有专门利用图片的局部空间结构。

本阶段的模型：

```text
Flatten → Linear(784, 128) → ReLU → Linear(128, 10)
```

| 步骤 | 输出 shape |
|---|---|
| 输入图片 | `[B, 1, 28, 28]` |
| Flatten | `[B, 784]` |
| 第一层 Linear | `[B, 128]` |
| ReLU | `[B, 128]` |
| 第二层 Linear | `[B, 10]` |

128 是人为设置的隐藏特征数量。每张图片计算出 128 个特征值，但所有图片共享同一套权重与偏置，并不是每张图片拥有自己的参数。

参数数量：

| 层 | 权重 | 偏置 | 合计 |
|---|---:|---:|---:|
| Linear(784, 128) | 784 × 128 | 128 | 100,480 |
| Linear(128, 10) | 128 × 10 | 10 | 1,290 |
| 总计 | | | **101,770** |

ReLU 将负值变成 0，保留正值，没有可训练参数。它提供非线性；如果只连续堆叠 Linear，中间没有非线性激活，整体仍能合并成一个线性变换。

## 4. Logits、预测类别与 CrossEntropyLoss

模型输出 `[B, 10]` 的 logits，也就是每张图片对应 10 个类别的原始分数。它们可以为负数，不要求总和为 1。

```python
prediction = model(images)
predicted_labels = prediction.argmax(dim=1)
loss = loss_api(prediction, labels)
```

`argmax(dim=1)` 在每张图片的 10 个分数中选出最大值的位置，得到 `[B]` 的预测类别。这里不需要先计算 softmax，因为 softmax 不会改变哪个分数最大。

使用 `nn.CrossEntropyLoss()` 时直接传入 logits 和真实类别：

| 参数 | Shape | dtype |
|---|---|---|
| prediction | `[B, 10]` | float32 |
| labels | `[B]` | int64 |

不要先做 sigmoid 或 softmax 再传入这个损失函数。如果只是希望展示各类别的概率，可以额外使用 `torch.softmax(prediction, dim=1)`。

### Loss 不是错误率

我曾经问：

> loss 是 0.5644，那损失率不是变成 56% 了吗？

不是。Accuracy 是分类正确比例，可以用百分比表示；交叉熵 loss 是对预测质量的惩罚，不是比例，也不限制在 `[0, 1]`。

对于单个样本，普通多分类交叉熵可以理解为真实类别预测概率的负对数：

```text
loss = -ln(真实类别的预测概率)
```

真实类别概率越低，惩罚通常越大。因此，即使两个模型都预测错误，loss 也可能不同；非常自信地预测错误，会受到更大的惩罚。

打印时使用：

```python
f"accuracy={accuracy:.2%}, loss={average_loss:.6f}"
```

## 5. Batch、Step 和 Epoch

我当时问：

> 训练一个 epoch，更新参数 1875 次，每次更新不是有 32 张图片吗？一张图片更新的参数不应该取决于我的层数设计吗？

这里要区分“参数数量”和“更新次数”。参数数量由模型结构决定，而一次 `optimizer.step()` 可以更新整套参数。在当前写法中，一批图片共同计算平均 loss，然后计算梯度，更新一次模型参数。

| 概念 | 当前训练方式中的含义 |
|---|---|
| Batch | 一批样本，通常为 32 张 |
| Step | 用一个 batch 计算梯度并更新一次参数 |
| Epoch | 完整遍历一次训练集 |

早期使用全部 60,000 张图片时，每轮有 1,875 个 batch。正式划分后：

| 数据部分 | 样本数 | Batch 数 | 最后一批 |
|---|---:|---:|---:|
| 训练 | 54,000 | 1,688 | 16 |
| 验证 | 6,000 | 188 | 16 |
| 测试 | 10,000 | 313 | 16 |

DataLoader 默认 `drop_last=False`，保留最后不足一批的数据，所以模型中的 B 不能写死为 32。验证和测试虽然也遍历 batch，但不执行参数更新。

一个 epoch 只是把训练集遍历了一遍，不代表模型已经学好了。每轮遍历 DataLoader，和在循环外取出一个 batch 后反复训练几千次，是两件不同的事。

## 6. 训练时到底发生了什么

每个训练 batch 的核心过程是：

```python
optimizer.zero_grad(set_to_none=True)
prediction = model(images)
loss = loss_api(prediction, labels)
loss.backward()
optimizer.step()
```

先清空之前的梯度，forward 得到当前预测，再计算 loss。`loss.backward()` 根据计算图计算参数梯度，`optimizer.step()` 根据梯度和优化器规则更新参数。

模型和 optimizer 在 epoch 循环外创建，才能让后面的训练延续已经更新过的参数。每个 batch 都重新 forward，计算的是当前参数下的新梯度。

我曾经把这个过程理解为“记住 loss”或“记住梯度”。更准确地说，在当前没有 momentum 的 SGD 中，跨训练步骤保留下来的学习结果主要是更新后的参数。梯度每批清零并重新计算，历史 loss 列表只是记录，不参与参数更新。Momentum 或 Adam 可以额外维护梯度的历史统计，但那是优化器的状态。

不清零梯度会导致梯度累加，并不自动意味着训练更好。有意进行梯度累积需要相应安排 loss 缩放与参数更新时机，当前实验不使用这种方式。

## 7. model.train()、model.eval() 和 no_grad()

| 操作 | 作用 |
|---|---|
| `model.train()` | 设置训练模式 |
| `model.eval()` | 设置评估模式 |
| `torch.no_grad()` | 暂时关闭梯度记录 |
| `loss.backward()` | 计算梯度 |
| `optimizer.step()` | 更新参数 |

train/eval 本身不会执行 forward，也不会自动计算梯度或更新参数。它们主要影响某些训练与评估行为不同的层：

- Dropout：训练时随机丢弃部分特征；评估时停止随机丢弃。
- BatchNorm：默认配置下，训练时使用当前 batch 统计量并更新运行统计量；评估时使用保存的运行统计量。

当前模型只有 Linear 和 ReLU，两种模式下计算行为相同，但仍明确设置模式。验证和测试通常同时使用 `eval()` 与 `no_grad()`，因为它们解决的是不同问题。

退出 `no_grad()` 后，梯度记录恢复到进入前的状态，但模型模式不会自动切回 train。下一轮训练前需要重新调用 `model.train()`。

加载模型时要对真正用于预测的实例调用：

```python
loaded_model.load_state_dict(state)
loaded_model.eval()
```

对旧的 `model` 调用 eval，不会改变新建的 `loaded_model`。

## 8. 如何统计 Accuracy 与平均 Loss

每轮训练和每轮验证都分别重置累计量，逐批统计：

```python
correct_num += (prediction.argmax(dim=1) == labels).sum().item()
total_num += labels.shape[0]
total_loss += loss.item() * labels.shape[0]
```

遍历结束后：

```python
accuracy = correct_num / total_num
average_loss = total_loss / total_num
```

当前 CrossEntropyLoss 默认返回 batch 平均 loss。先乘当前 batch 的实际样本数，再除以全部样本数，可以让最后不足 32 张的 batch 得到正确权重。

不要把可反向传播的 `loss` 和累计量混用。`loss` 用于当前 batch 的反向传播，`total_loss` 是保存统计结果的 Python 数值。

### 训练一定要统计指标吗？

训练必须计算用于 backward 的 loss，但累计整个 epoch 的 loss 和 accuracy 是可选的。保留训练与验证两套指标，是为了比较“训练数据学得怎么样”和“未参与参数更新的数据表现怎么样”。

本脚本的训练指标来自不断更新中的模型，每批统计的是更新前的预测；验证指标来自该 epoch 训练结束后固定参数的模型。因此，两者不是完全相同的评估口径。第一轮验证准确率高于训练过程的累计准确率，并不奇怪。

## 9. 验证、测试与最佳 Checkpoint

我当时问：

> 验证理论上是不是很像 inference？

是的。Inference（推理）只需要输入并得到预测；验证在推理之外，还使用真实标签计算 loss 和 accuracy。验证标签不用于 backward，但验证结果参与选择模型，所以测试集需要另行保留。

每个 epoch 的顺序是：

1. 训练：更新参数并统计训练指标。
2. 验证：固定参数，计算验证指标。
3. 比较：验证准确率是否超过历史最佳。
4. 保存：如果超过，覆盖最佳 checkpoint。

```python
# epoch 循环外
best_val_acc = -1.0
best_epoch = 0

# 每轮验证结束后
if accuracy_curr > best_val_acc:
    best_val_acc = accuracy_curr
    best_epoch = epoch + 1
    torch.save(model.state_dict(), checkpoint_path)
```

最佳记录必须跨 epoch 保留，不能每轮重置。判断以后也必须更新 `best_val_acc`，否则每一轮都会覆盖文件。这里使用严格的 `>`，实际准确率相同时保留先前模型。

本实验只按验证准确率选择模型，并没有再用验证 loss 处理平局。打印百分比经过四舍五入，显示相同也不一定代表内部数值完全相同。

所有训练完成后，创建新的同结构模型、加载最佳权重，再评估官方测试集。最后一轮模型不一定是最好的一轮，所以不能直接默认使用训练结束时内存中的参数。

### 保存的是什么？

`torch.save(model.state_dict(), path)` 在当前模型中保存各层的权重和偏置，没有保存完整的 Python 类定义、训练循环或 optimizer 状态。恢复时需要先创建相同结构的模型，再加载参数。

`.pt` 是常用的 PyTorch 文件扩展名，不能只靠扩展名判断文件内容；本实验明确保存的是 state_dict。若要完整恢复训练，还可以另外保存 optimizer 状态、epoch 等信息，本阶段没有实现训练断点续训。

## 10. 实验结果

### Notebook 中选模与测试

| 指标 | 结果 |
|---|---:|
| 最佳 epoch | 5 |
| 最佳验证准确率 | 87.25% |
| 对应验证 loss | 约 0.3466 |
| 测试样本数 | 10,000 |
| 测试准确率 | 86.75% |
| 测试平均 loss | 0.366023 |

重新加载最佳参数后，对完整 6,000 条验证数据再次评估，指标与保存时一致。

### 独立 Python 脚本

脚本：`scripts/stage_b_fashion_mnist.py`

配置：SGD，学习率 0.1，batch_size=32，5 epochs，CPU。

| Epoch | Train loss | Train accuracy | Val loss | Val accuracy |
|---|---:|---:|---:|---:|
| 1 | 0.5760 | 79.54% | 0.4322 | 84.65% |
| 2 | 0.4129 | 85.14% | 0.3823 | 85.88% |
| 3 | 0.3722 | 86.59% | 0.3810 | 86.02% |
| 4 | 0.3472 | 87.34% | 0.3823 | 86.02% |
| 5 | 0.3304 | 87.84% | 0.3544 | 86.80% |

| 最终指标 | 结果 |
|---|---:|
| 最佳 epoch | 5 |
| 最佳验证准确率 | 86.80% |
| 测试样本数 | 10,000 |
| 测试准确率 | **86.41%** |
| 测试平均 loss | **0.373224** |

脚本保存位置为项目根目录下：

```text
checkpoints/stageb_fashion_mnist_best.pt
```

脚本通过 `Path(__file__).resolve().parent.parent` 定位项目根目录，使数据和 checkpoint 路径不依赖启动命令时所在的目录。

两次运行是分别训练的结果，不能拼成同一次实验。数据划分固定，但初始化和训练顺序没有全部固定，因此指标存在差异。两次测试用于阶段记录，没有根据测试结果继续调参。

### 预测图片观察

展示了按遍历顺序遇到的前 6 个正确案例和前 6 个错误案例。错误示例包括 Shirt 被识别为 Coat、Pullover 或 T-shirt/top，Dress 被识别为 Shirt，以及 Sandal 被识别为 Ankle boot。

这些图片帮助理解模型具体输出，但不能代表总体错误分布，也不能仅凭几个案例确定错误原因。图片分辨率低、类别外形相似都是需要考虑的因素，本阶段不开展大量错误审查。

## 11. 为什么训练改善，验证却可能波动？

每次参数更新直接优化的是当前训练 batch 的 loss，并不保证验证集表现同步改善。训练 loss 下降说明模型更适应训练数据，但最终希望它也能处理未参与训练的数据，这就是 generalization（泛化能力）。

一次验证 loss 上升，不能直接证明模型过拟合。可能是正常波动或优化过程中的变化，需要结合多轮趋势判断。如果训练表现持续改善，而验证表现长期变差，才更值得怀疑 overfitting（过拟合）。

验证集固定时，“这一轮验证不佳”仍然是当前模型在这批固定样本上的实际表现，不能因为训练指标改善就忽略它。

## 12. 如何根据结果考虑调参？

不能从“某张衬衫识别错了”直接推导出“应该增加两层”。错误图片提供线索，训练与验证表现帮助判断改动方向。

| 观察现象 | 可能的判断与下一步 |
|---|---|
| 训练和验证都不理想，训练 loss 仍在下降 | 可能尚未训练充分，先考虑增加轮数 |
| 训练表现较差且长期停滞 | 先检查实现与学习率，再考虑模型容量 |
| 训练表现很好，验证明显落后 | 可能过拟合，可考虑正则化、数据增强或减小模型 |
| Loss 大幅波动或持续上升 | 检查代码、数据和学习率 |
| 特定类别频繁混淆 | 查看混淆矩阵和代表样本，再判断原因 |

这些是诊断方向，不是看到某种现象就必然对应某个原因。如果需要严格比较训练与验证表现，可以用同一个固定模型在 eval 模式下分别评估。

### 加宽与加深有什么区别？

把 `784 → 128 → 10` 改成 `784 → 256 → 10` 是加宽，让每张图片得到更多隐藏特征。我们设置特征数量，具体特征由训练得到的权重决定，不一定能直接解释为“袖子”或“领口”。

改成 `784 → 128 → 64 → 10`，并在隐藏层之间加入激活函数，是加深，让后面的层继续组合前面的特征，进行多阶段非线性变换。

两者都不保证效果提升：模型可能更难训练，或者更容易过拟合。调参时先明确要回答的问题，尽量一次改变一个主要因素，再通过多个随机种子确认有意义的结果。验证集用于选方案，测试集用于最终评估。

本阶段只理解这些思路，没有追加调参实验。

## 13. MLP、CNN、Transformer 与 Decoder

我当时总结：

> 目前我们用的本质上都是 MLP 感知机的东西，并没有用上卷积，例如 CNN。

这个理解正确。本模型是 MLP（Multilayer Perceptron，多层感知机），由全连接层和激活函数组成。

| 概念 | 主要特点 |
|---|---|
| MLP | 用全连接层和激活函数变换特征 |
| CNN，卷积神经网络 | 用共享的卷积核处理局部区域，利用空间结构 |
| Transformer | 用 Attention 进行位置之间的信息交互，并通过 FFN 等模块处理特征 |

这些架构可以组合。Transformer 内部的 FFN 通常就是一个 MLP，例如之前拆解 BERT 时观察到的 `768 → 3072 → 768`。

CNN 不只是“更多层的 MLP”。它在计算方式上利用了局部连接和权重共享，因此结构是否适合数据，与参数数量同样重要。

我提到的“autodecoder”需要区分两个概念：

- **Autoencoder，自编码器**：先编码，再解码以重建输入。它可以通过 MLP、CNN 或 Transformer 实现。
- **Autoregressive decoder，自回归解码器**：根据已经出现的内容，逐步预测下一个 token。GPT 类模型通常使用 decoder-only Transformer。

“预测什么、怎样训练”与“模型内部采用什么架构”是不同角度。先理解预测下一个字符的任务，再学习 Transformer，是后续 Stage C 到 CS336 的衔接。

## 14. Stage B 完成状态

已经完成数据加载与划分、mini-batch 训练、多分类损失与指标统计、独立验证、最佳 checkpoint 保存与恢复、测试集评估、预测可视化，以及独立 Python 程序。

下一阶段开始字符级语言模型：理解字符与 ID 的映射、输入与目标错位、下一个字符预测，以及如何逐步生成文本。继续沿用已经学会的 forward、loss、backward、optimizer 和评估流程。