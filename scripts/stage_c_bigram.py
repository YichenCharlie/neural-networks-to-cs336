import torch
import torch.nn as nn


text = "hello world"
unsorted_char = set()

for char in text:
    unsorted_char.add(char)
chars = sorted(unsorted_char)
world_len = len(unsorted_char)
itos = {}
stoi = {}

for i, char in enumerate(chars):
    itos[i] = char
    stoi[char] = i
encode = []

for char in text:
    encode.append(stoi[char])

print(world_len, "".join(itos[i] for i in encode) == text)

table = nn.Embedding(world_len,world_len)
optimizer = torch.optim.SGD(table.parameters() , lr=0.1)
loss_api = nn.CrossEntropyLoss()
B , T = 2, 4

full_x = torch.tensor(encode[:-1], dtype=torch.long)
full_y = torch.tensor(encode[1:], dtype=torch.long)

with torch.no_grad():
    print("Initial full-text loss:", loss_api(table(full_x), full_y).item())



for step in range(1000):
    x1 = []
    y1 = []
    starts = torch.randint(0, len(encode) - T, size=(B,))
    for start in starts:
        window = encode[start:start + T + 1]
        x1.append(torch.tensor(window[:-1] ,dtype=torch.long))
        y1.append(torch.tensor(window[1:] , dtype=torch.long))

    x_batch = torch.stack(x1)
    y_batch = torch.stack(y1)

    optimizer.zero_grad(set_to_none=True)
    pred = table(x_batch)

    flat_logit = pred.reshape(-1,8)
    flat_target = y_batch.reshape(-1)
    loss = loss_api(flat_logit,flat_target)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        if (step + 1) % 200 == 0:
            print(f"Step {step + 1}: batch_loss={loss.item():.4f}")

with torch.no_grad():
    print("Final full-text loss:", loss_api(table(full_x), full_y).item())


with torch.no_grad():
    new_text = []
    output_code = []
    for char in text[:-1]:
         output = (table.weight[stoi[char]])
         output_code.append(output.argmax().item())

    decode = []
    for i in output_code:
        decode.append(itos[i])

    print(decode)


curr = "h"
res =["h"]
with torch.no_grad():
    for step in range(20):
        logits = table.weight[stoi[curr]]
        hi_prob = torch.softmax(logits,dim=0)
        nxt_id = torch.multinomial(hi_prob,num_samples=1)
        res.append(itos[nxt_id.item()])
        curr = res[-1]

print("Generated:", repr("".join(res)))
print("Length:", len(res))