### Chapter 3. 선형 회귀 분석

#### 3.1 선형회귀분석이란 무엇인가

1. 선형회귀분석이란?

   주어진 데이터를 가장 잘 설명하는 직선 하나를 찾는 것이다.

   - 단순선형회귀(Simple Linear Regression): 하나의 독립변수에 대하여 선형회귀분석을 하는 것이다.
   - 다중선형회귀(Multivariate Linear Regression): 독립변수가 여러 개인 경우에 대하여 선형회귀분석을 하는 것이다.

   단순선형회귀분석을 한다는 것은 x와 y라는 데이터가 주어졌을 때, y = w*x + b라는 직선의 방적식에서 데이터를 가장 잘 표현하는 변수 w와 b를 찾는다는 뜻이다. 이 때, w와 b는 가중치(Weight)와 편차(Bias)를 각각 한 문자로 표현한 것이다.

   

#### 3.2 손실 함수 및 경사하강법

1. 평균제곱오차(Mean Squared Error)

   어떤 w, b 쌍에 대해서 데이터와 얼마나 잘 맞는지 수치적으로 계산을 할 수 있어야 하는데, 이때 사용되는 척도 중 대표적인 것이 평균제곱오차(MSE)이다. n개의 예측값 있고 주어진 데이터 값을 y라 할 때, 평균 제곱오차 식은 다음과 같다. 

   

   ***... 수식이 많으므로 생략 ...***

   

---

#### 3.3 파이토치에서의 경사하강법

​	파이토치에서는 데이터의 기본 단위로 **텐서(Tensor)**라는 것을 사용한다. 텐서는 다차원 배열(array)이라고 정의할 수 있다. 즉 텐서는 n차우너의 배열을 전부 포함하는 넓은 개념이고 파이토치는 이러한 텐서를 기본 연산의 단위로 사용한다. 

```python
import torch					# 프레임 워크 불러오기
X = torch.Tensor(2,3)			# X라는 변수에 파이토치 텐서를 임의의 난수를 가진 배열 생성 (2x3)
```

​	텐서를 생성하면서 원하는 값으로 초기화 하려면 인수로 배열을 전달해야한다.

```python
X = torch.tensor([1,2,3], [4,5,6])
```

​	torch.tensor 함수는 인수로 data, dtype, device, requires_grad 등을 받는다.  dtype에는 데이터를 저장할 자료형이 들어간다. 자료형은 다음표와 같이 다양한데, 기본값은 FloatTensor이다. 또한, GPU용 텐서 자료형도 지원한다.

| 자료형                 | CPU텐서            | GPU텐서                 |
| ---------------------- | ------------------ | ----------------------- |
| 32비트 부동소수점      | torch.FloatTensor  | torch.cuda.FloatTensor  |
| 64비트 부동소수점      | torch.DoubleTensor | torch.cuda.DoubleTensor |
| 16비트 부동소수점      | torch.DobuleTensor | torch.cuda.HalfTensor   |
| 8비트 정수(부호 없음)  | torch.ByteTensor   | torch.cuda.ByteTensor   |
| 8비트 정수(부호 있음)  | torch.CharTensor   | torch.cuda.CharTensor   |
| 16비트 정수(부호 있음) | torch.ShortTensor  | torch.cuda.ShortTensor  |
| 32비트 정수(부호 있음) | torch.IntTensor    | torch.cuda.IntTensor    |
| 64비트 정수(부호 있음) | torch.LongTensor   | torch.cuda.LongTensor   |

​	이어서 device는 이 텐서를 어느 기기에 올릴 것인지를 명시한다. 마지막으로 requires_grad는 이 텐서에 대한 기울기를 저장할지 여부를 지정한다. 기본값은 False인데 명시적으로 지정하려면 예를 들어 다음과 같이 쓰면 됩니다.

```python
x_tensor = torch.tensor(data=[2.0,3.0], requires_grad=True)
```

​	이렇게 생성한 텐서를 가지고 연산 그래프를 생성하면 연산 그래프는 어떠한 결과를 가져올 것입니다. 우선 기울기를 계산하는 코드 예는 다음과 같다.

```python
import torch

x = torch.tensor(data=[2.0,3.0], requires_grad=True)
y = x**2
z = 2*y + 3

target = torch.tensor([3.0,4.0])
loss = torch.sum(torch.abs(z-target))
loss.backward()

print(x.grad, y.grad, z.grad)
```

​	먼저 x라는 텐서를 생성하며 기울기를 계산하도록 지정햇고, 따라서 z라는 변수에 연산 그래프의 결괏값이 저장된다. z와 목표값인 target의 절대값의 차이를 계산하고 torch.sum()이란 함수를 통해 3X4 모양이었던 두 값의 차이를 숫자 하나로 바꾼다. 그 다음 loss.backward()함수를 호출하면 연산 그래프를 쭉 따라가면서 잎 노드(Leaf node) x에 대한 기울기를 계산한다. **여기서 잎 노드는 다른 변수를 통해 계산되는 y나 z가 아니라 그 자체가 값이 x 같은 노드를 의미 한다.** 









