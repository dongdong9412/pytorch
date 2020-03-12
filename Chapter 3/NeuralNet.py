import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt

num_data = 1000                                                     # x data의 개수
num_epoch = 10000                                                   # training 횟수

noise = init.normal_(torch.FloatTensor(num_data, 1), std=1)         # 정규분포를 따르는 num_data x 1의 크기의 랜덤 배열 생성
x = init.uniform_(torch.Tensor(num_data, 1), -15, 15)               # -15에서 15사이의 num_data x 1의 크기의 랜덤 배열 생성
y = (x**2) + 3                                                      # 찾고자 하는 label
y_noise = y + noise                                                 # training 시키고자 하는 model


model = nn.Sequential(                                              # 괄호 안의 함수를 순차적으로 실행
    nn.Linear(1, 6),                                                # Input 1 Output 6의 선형 회귀
    nn.ReLU(),                                                      # 활성화 함수
    nn.Linear(6, 10),
    nn.ReLU(),
    nn.Linear(10, 6),
    nn.ReLU(),
    nn.Linear(6, 1),                                                # 은닉계층을 지나 최종적으로는 Output이 1이어야함
)

loss_func = nn.L1Loss()                                             # 손실함수
optimizer = optim.SGD(model.parameters(), lr=0.002)                 # 최적화 기법. 학습률 0.002

loss_array = []                                                     # 각 step별 오차율을 저장 할 배열
for i in range(num_epoch):
    optimizer.zero_grad()                                           # 기울기 초기화
    output = model(x)                                               # model에 x값을 넣음
    loss = loss_func(output, y_noise)                               # 학습 model(output)과 y_noise의 오차율을 계산
    loss.backward()                                                 # 기울기 계산
    optimizer.step()                                                # 학습률 만큼 현재 기울기 업데이트

    loss_array.append(loss)                                         # 현재 오차율과 step을 저장
    if i % 10:
        print(loss.data)

plt.plot(loss_array)
plt.show()


