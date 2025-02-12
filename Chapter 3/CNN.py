import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

batch_size = 256                                                                            # 한번에 묶을 data size
learning_rate = 0.0002                                                                      # 학습률
num_epoch = 10                                                                              # training 횟수

class CNN(nn.Module):
    def __init__(self):                                                                     # Class 초기화 함수 선언
        super(CNN, self).__init__()                                                         # super 클래스는 CNN 클래스의 부모 클래스인 nn.Module을 초기화 하는 역할
        self.layer = nn.Sequential(                                                         # layer 함수는 다음 동작을 연속적으로 실행
            nn.Conv2d(1, 16, 5),                                                            # 합성곱을 1개의 input에 16개의 output으로 만들며 커널은 5x5 (스트라이드는 1, 패딩은 0이 기본값)
            nn.ReLU(),                                                                      # 활성화 함수
            nn.Conv2d(16, 32, 5),                                                           # 합성곱을 16개의 input에 32개의 output으로 만들며 커널은 5x5
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                                             # kernel size는 2, stride도 2. 크기가 반으로 줄어듬
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*3*3, 100),                                                         # 선형 회귀 (input=64*3*3, output=100)
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):                                                                   # 종합 함수 수행
        out = self.layer(x)
        out = out.view(batch_size, -1)                                                      # [batch size, 자동 맞춤]
        out = self.fc_layer(out)
        return out

def main():
    mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(),
                         target_transform=None, download=True)                              # MNIST에서 학습할 데이터를 파이토치 텐서 형태로 가져옴. (경로에 데이터가 없을 경우 다운로드)
    mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(),
                        target_transform=None, download=True)                               # MNIST에서 테스트할 데이터를 파이토치 텐서 형태로 가져옴. (경로에 데이터가 없을 경우 다운로드)

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                           shuffle=True, num_workers=1, drop_last=True)     # MNIST에서 가져온 학습할 데이터를 batch size만큼 묶고 순서를 섞는다. 사용 프로세스는 2개에 남는 데이터는 버림.
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,
                                          shuffle=False, num_workers=1, drop_last=True)     # MNIST에서 가져온 테스트할 데이터를 batch size만큼 묶고 순서대로 나열한다. 사용 프로세스는 2개에 남는 데이터는 버림


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_arr = []
    for i in range(num_epoch):
        for j,[image, label] in enumerate(train_loader):
            x = image.to(device)
            y_ = label.to(device)

            output = model.forward(x)
            loss = loss_func(output, y_)
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()

            if j % 1000 == 0:
                print(loss)
                loss_arr.append(loss.cpu().detach().numpy())

    correct = 0
    total = 0

    with torch.no_grad():
        for image, label in test_loader:
            x = image.to(device)
            y_ = label.to(device)

            output = model.forward(x)
            _, output_index = torch.max(output, 1)

            total += label.size(0)
            correct += (output_index == y_).sum().float()

        print("Accuracy of Test Data: {}".format(100*correct/total))

if __name__ == '__main__':
    main()