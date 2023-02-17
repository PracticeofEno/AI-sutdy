import torch
import torch.nn as nn

# 배치 크기 × 채널 × 높이(height) × 너비(widht)의 크기의 텐서를 선언
inputs = torch.Tensor(1, 1, 28, 28)
print('텐서의 크기 : {}'.format(inputs.shape))

# 1채널 입력, 32채널 output, 3의 kernel사이즈, 페딩 1
conv1 = nn.Conv2d(1, 32, 3, padding=1)
print(conv1)

# 32채널 input, 64채널 output, 커널사이즈 3, 페딩 1
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
print(conv2)

# 맥스풀링. 정수 하나를 인자로 넣으면 커널 사이즈와 스트라이드가 둘 다 해당값으로 지정
pool = nn.MaxPool2d(2)
print(pool)

out = conv1(inputs)
print(out.shape)

out = pool(out)
print(out.shape)

out = conv2(out)
print(out.shape)

out = pool(out)
print(out.shape)

# 첫번째 차원인 배치 차원은 그대로 두고 나머지는 펼쳐라
out = out.view(out.size(0), -1) 
print(out.shape)

fc = nn.Linear(3136, 10) # input_dim = 3,136, output_dim = 10
out = fc(out)
print(out.shape)