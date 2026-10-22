import torch
import torch.nn as nn
from pathlib import Path


class XORmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2,4)
        self.layer2 = nn.Linear(4,1)

    def forward(self,x):
        y1 = self.layer1(x)
        y2 = torch.relu(y1)
        y3 = self.layer2(y2)
        return y3

x = torch.tensor([
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1],
], dtype=torch.float32)

y = torch.tensor([
    [0],
    [1],
    [1],
    [0],
], dtype=torch.float32)

model = XORmodel()
optimizer = torch.optim.SGD(model.parameters(),lr= 0.1)
loss_api = nn.BCEWithLogitsLoss()

for step in range(3000):
    optimizer.zero_grad(set_to_none=True)
    y_infer = model(x)
    loss = loss_api(y_infer,y)
    loss.backward()
    optimizer.step()

    if (step + 1) % 500 == 0:
        with torch.no_grad():
            y_infer = model(x)
            loss = loss_api(y_infer,y)
            print(f"Step {step + 1}: loss={loss.item():.6f}")
            print("Probabilities:", torch.sigmoid(y_infer))
            print("Predictions:", (torch.sigmoid(y_infer) >= 0.5).long())

Path("checkpoints").mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(),'checkpoints/stage_a_final.pt')
state = torch.load(
    "checkpoints/stage_a_final.pt",
    map_location="cpu",
    weights_only=True,
)
loaded_model = XORmodel()
loaded_model.load_state_dict(state)

with torch.no_grad():
    output1 = model(x)
    output2 = loaded_model(x)

print(torch.allclose(output1,output2))