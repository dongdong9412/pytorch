import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

num_data = 1000                                                 # x data의 개수
num_epoch = 5000                                                # 학습 횟수

x = init.uniform_(torch.Tensor(num_data,1),-10,10)              # -10 ~ 10의 랜덤 배열 생성 (num_data 개수만큼)
noise = init.normal_(torch.FloatTensor(num_data,1), std=1)      # num_data 개수만큼 정규분포 값을 따르는 배열 생성

y = 2*x + 3
y_noise = 2*(x+noise) + 3

model = nn.Linear(1,1)                                          # 
loss_func = nn.L1Loss()

optimizer = optim.SGD(model.parameters(),lr=0.01)

label = y_noise
for i in range(num_epoch):
    optimizer.zero_grad()
    output = model(x)

    loss = loss_func(output,label)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(loss.data)

param_list = list(model.parameters())
print(param_list[0].item(), param_list[1].item())
