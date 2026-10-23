# Stage B — 从数据载入到第一个 Epoch

## 目前的主线

自己写懂图片分类的完整流程：数据载入、组织 batch、模型 forward、计算 loss、反向传播、更新参数，之后再加入验证。

目前不开展 batch size 调参或性能优化实验。

## 按我的理解，把流程串起来

数据 → 数据载入 → 确定批次大小 batch_size → 取出一批数据 → 数据平摊，把每张图片摊开 → 模型预测 → 计算 loss → 计算梯度 → 更新参数。

这里要区分两个操作：

- DataLoader：把多条样本组成一批。
- Flatten：把每张图片的像素展开成一行，保留 batch 维度。

组批和展平不是同一件事。

## 1. 数据与数据载入

这次使用 Fashion-MNIST，是服装图片数据集，不是手写数字 MNIST。

每张图片是 28 × 28 的灰度图，共有 10 个类别。标签是类别编号，例如 9 表示 Ankle boot（短靴），不是图片里写着数字 9。

通过 Dataset 取一条样本：

    image, label = train_dataset[0]

得到的是一张图片和一个标签。使用 ToTensor 后，图片 shape 为 [1, 28, 28]，dtype 为 float32，像素值范围为 [0, 1]。

这里的 1 是灰度通道数。

## 2. 确定批次大小：batch_size

我当时问：

> 我数据集里的照片不应该是 [1,28,28] 吗，哪来的 32 批次，批次不是靠 DataLoader 干出来的吗？

单张图片确实是 [1, 28, 28]。前面的 32，就是 DataLoader 把 32 张图片组到一起后产生的。

    images, labels = next(iter(loaded_data))

这行取出的是一整批数据：

| 对象 | Shape | 含义 |
|---|---|---|
| image | [1, 28, 28] | 单张图片 |
| images | [32, 1, 28, 28] | 一批 32 张图片 |
| labels | [32] | 这 32 张图片对应的类别编号 |

32 表示一批有 32 张图片，不是有 32 个批次。标签的 dtype 是 int64。

Dataset 提供单条样本，DataLoader 负责组批；取出来的 images 已经是组好的图片 Tensor。

## 3. 数据平摊：将每张图片摊开

为了送入 MLP，把每张图片的像素按顺序展开：

    flat_images = images.flatten(start_dim=1)

Shape 变化：

    [32, 1, 28, 28] → [32, 784]

保留第 0 维的 32，从第 1 维开始合并后面的维度。

Flatten 不修改像素数值，没有可训练参数，也不会把原数据集改掉。展平前后的元素总数都是：

    32 × 1 × 28 × 28 = 25,088

不能对 DataLoader 调用 flatten；需要对它取出的图片 Tensor 调用。

## 4. 模型如何接收这些数据

本次模型：

    Linear(784, 128) → ReLU → Linear(128, 10)

| 步骤 | Shape |
|---|---|
| 输入图片 | [B, 1, 28, 28] |
| 展平 | [B, 784] |
| 第一层 Linear | [B, 128] |
| ReLU | [B, 128] |
| 第二层 Linear | [B, 10] |

B 表示当前批次的样本数，不写死为 32。

128 是人为选择的隐藏特征数量，不是每条样本独享的 128 个参数。所有样本共享同一套模型参数。

最后的 10 对应 10 个类别，输出的是 logits，即未经概率转换的分类分数。

使用 CrossEntropyLoss 时，直接传入：

- 模型 logits：[B, 10]
- 真实标签：[B]，int64 类型的类别编号

不需要先对 logits 做 sigmoid 或 softmax。

## 5. Batch、Step 和 Epoch

我当时问：

> 训练一个 epoch，更新参数 1875 次，每次更新不是有 32 张图片吗？一张图片更新的参数不应该取决于我的层数设计吗？

这里混淆了“参数数量”和“更新次数”。

- 参数数量取决于模型结构。
- 更新次数指执行了多少次 optimizer.step()。
- 一次 step 可以更新模型中的很多个参数。

当前训练方式中，一批 32 张图片共同计算一个平均 loss。backward 根据这个 loss 计算各参数的梯度，optimizer.step() 再更新一次整套可训练参数。

不是每张图片分别拥有一套参数，也不是每张图片都单独调用一次 optimizer.step()。

| 概念 | 当前设置下的含义 |
|---|---|
| Batch | 一组 32 张图片 |
| Step | 用一批数据计算梯度，更新一次参数 |
| Epoch | 把整个训练集遍历一次 |

60,000 张训练图片，每批 32 张：

    60,000 ÷ 32 = 1,875 个 batch

因此，当前每批更新一次参数的写法中，一个 epoch 包含 1,875 次参数更新。

遍历完一个 epoch，不代表模型已经学好了，只代表每条训练样本在这一轮参与了一次训练。

## 6. 为什么循环会自动取下一批

我当时问：

> 这个执行一次，不是训练一个批次 32 张图片吗？那怎么让它训练第二个批次，为什么再进循环就会从这个批次到下个批次了？

    for batch_index, (images, labels) in enumerate(loaded_data):

for 会持续从同一个迭代器取下一批，直到这一轮数据用完。每次变量名还是 images、labels，但里面的数据已经换成下一批。

enumerate 只负责附上批次编号，不负责训练或切换数据。

之前的写法是在循环外只取一次 batch，然后循环 5,000 次，因此是在反复训练同一批 32 张图片。

完整训练则是：

- 外层循环控制 epoch 数量。
- 内层循环遍历 DataLoader。
- 每一批执行 forward、loss、backward 和 optimizer.step()。

模型和 optimizer 在循环外创建，让每次更新延续之前学到的参数。

## 7. 不能整除 batch_size 怎么办

我当时说“如果一个数据集是非整数”，实际指的是：

> 数据集的样本数除以 batch_size，不能整除怎么办？

默认会保留最后不足一批的样本。例如 100 张图片，batch_size=32，会分成：

    32、32、32、4

这也是模型不能把 batch size 写死的原因。

DataLoader 默认 drop_last=False；设置为 True 才会丢掉最后不足一批的数据。目前保留默认值即可。

batch size 更大，一个 epoch 的 step 数会减少，但每个 step 处理的数据更多，内存需求也更高。批次数少不等于更快学好，不需要为了减少 batch 数盲目增大 batch_size。

## 8. model.train() 是什么

model.train() 把模型切换到训练模式，本身不会计算梯度或更新参数。

- model.train() / model.eval()：控制某些层在训练和评估时的行为。
- torch.no_grad()：关闭梯度记录。
- loss.backward()：计算梯度。
- optimizer.step()：更新参数。

当前模型只有 Linear 和 ReLU，它们在 train/eval 模式下的计算行为相同。以后遇到 Dropout 等层，这个区别就会影响输出。

model.eval() 不等于关闭梯度。

## 9. 本次实际结果

已经完成一个 epoch：

    Epoch 1/1, Batch 300/1875, loss=0.5545
    Epoch 1/1, Batch 600/1875, loss=0.2990
    Epoch 1/1, Batch 900/1875, loss=0.4311
    Epoch 1/1, Batch 1200/1875, loss=0.4383
    Epoch 1/1, Batch 1500/1875, loss=0.4214
    Epoch 1/1, Batch 1800/1875, loss=0.3606
    Epoch 1/1, Batch 1875/1875, loss=0.5117

这些是所打印批次在参数更新前的 loss，不是整个 epoch 的平均 loss。不同批次的图片难度不同，loss 不要求每次都下降。

目前只确认训练循环可以完整运行，还没有计算整个 epoch 的平均 loss、accuracy，也没有评估独立验证集。

一个 epoch 遍历完整个训练集；每个 step 处理一个 batch，并更新一次模型参数。