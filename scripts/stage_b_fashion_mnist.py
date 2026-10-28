import torch
import torch.nn as nn
from torchvision import datasets,transforms
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import random_split

ROOT = Path(__file__).resolve().parent.parent

train_dataset = datasets.FashionMNIST(
    root= ROOT / "data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

train_subset, val_subset = random_split(
    train_dataset,
    [54000, 6000],
    generator=torch.Generator().manual_seed(42),
)

train_set = DataLoader(train_subset,batch_size=32,shuffle=True,num_workers= 0)
val_set = DataLoader(val_subset,batch_size=32,shuffle=False,num_workers= 0)

checkpoint_dir = ROOT / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = checkpoint_dir / "stageb_fashion_mnist_best.pt"


class classiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784,128)
        self.layer2 = nn.Linear(128,10)

    def forward(self,images):
        input = images.flatten(start_dim = 1)
        y1 = self.layer1(input)
        y2 = torch.relu(y1)
        y3 = self.layer2(y2)

        return y3

model = classiModel()
optimizer = torch.optim.SGD(model.parameters(),lr= 0.1)
loss_api = nn.CrossEntropyLoss()

epochs = 5
best_val_acc = -1.0
best_epoch = 0

for epoch in range(epochs):
    #val的参数计算
    total_num = 0
    corr_num = 0
    total_loss = 0


    #train的参数计算
    train_total_num = 0
    train_corr_num = 0
    train_total_loss = 0

    model.train()

    for images,labels in train_set:

        #训练部分
        optimizer.zero_grad(set_to_none=True)
        the_labels = model(images)
        loss = loss_api(the_labels,labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
        

            train_labels = the_labels.argmax(dim = 1)
            train_corr_num += (train_labels == labels).sum().item()
            train_total_num += labels.shape[0]
            train_total_loss += loss.item() * labels.shape[0]

    train_acc = train_corr_num / train_total_num
    train_av_loss = train_total_loss / train_total_num



    print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"train_loss={train_av_loss:.4f}, train_acc={train_acc:.2%} | "
                
            )

    model.eval()

    for images,labels in val_set:
        with torch.no_grad():
            val_pred = model(images)
            loss = loss_api(val_pred,labels)

            #计算各项参数部分
            val_labels = val_pred.argmax(dim = 1)
            corr_num += (val_labels == labels).sum().item()
            total_num += labels.shape[0]
            total_loss += loss.item() * labels.shape[0]

    accuracy_curr = corr_num / total_num
    av_loss = total_loss / total_num

    #保存最佳轮次参数
    if accuracy_curr > best_val_acc:
        best_val_acc = accuracy_curr
        best_epoch = epoch + 1
        torch.save(model.state_dict(), checkpoint_path)

    #打印每轮结果
    print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"val_loss={av_loss:.4f}, val_acc={accuracy_curr:.2%} | "
        )

#测试阶段

test_dataset = datasets.FashionMNIST(
    root=ROOT / "data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

test_loader = DataLoader(test_dataset,batch_size=32,shuffle= False,num_workers= 0 )


loaded_model = classiModel()
infer_para = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)
loaded_model.load_state_dict(infer_para)
loaded_model.eval()

test_total_num = 0
test_correct_num = 0
test_total_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        prediction_infer = loaded_model(images)
        loss = loss_api(prediction_infer, labels)

        the_infer_labels = prediction_infer.argmax(dim=1)
        test_correct_num += (the_infer_labels == labels).sum().item()
        test_total_num += labels.shape[0]
        test_total_loss += loss.item() * labels.shape[0]

av_test_loss = test_total_loss / test_total_num
accuracy = test_correct_num / test_total_num

print(f"Best epoch: {best_epoch}, best val accuracy: {best_val_acc:.2%}")
print("Test samples:", test_total_num)
print(f"Test accuracy: {accuracy:.2%}")
print(f"Test average loss: {av_test_loss:.6f}")
