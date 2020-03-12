import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

num_data = 1000                                                 # x data의 개수
num_epoch = 5000                                                # training 횟수

x = init.uniform_(torch.Tensor(num_data,1),-10,10)              # x에 100x1 크기의 -10에서 10까지의 랜덤 값 생성
noise = init.normal_(torch.FloatTensor(num_data,1), std=1)      # x의 외란으로 작용할 100x1 크기의 정규분포를 따르는 랜덤 값 생성

y = 2*x + 3
y_noise = 2*(x+noise) + 3

model = nn.Linear(1,1)                                          # 1대 1 대응의 선형회귀
loss_func = nn.L1Loss()                                         # L1 Loss Function (차이의 절대값 평균)

optimizer = optim.SGD(model.parameters(),lr=0.01)               # SGD(Stochastic Gradient Descent) 경사 하강법

label = y_noise
for i in range(num_epoch):
    optimizer.zero_grad()                                       # 경사하강법을 위한 기울기 초기화
    output = model(x)

    loss = loss_func(output,label)                              # 손실 함수에 output과 label을 전달
    loss.backward()                                             # x에 대한 기울기 계산
    optimizer.step()                                            # 기울기에 대한 학습률을 곱하여 업데이트

    if i % 10 == 0:
        print(loss.data)

param_list = list(model.parameters())
print(param_list[0].item(), param_list[1].item())
