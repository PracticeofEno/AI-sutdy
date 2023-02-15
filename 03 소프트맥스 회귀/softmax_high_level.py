import torch
import torch.nn.functional as F

torch.manual_seed(1)

z = torch.FloatTensor([1, 2, 3])

hypothesis = F.softmax(z, dim=0)

z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)

y = torch.randint(5, (3,)).long()

# 모든 원소가 0의 값을 가진 3 × 5 텐서 생성
y_one_hot = torch.zeros_like(hypothesis) 
y_one_hot.scatter_(1, y.unsqueeze(1), 1)


## Cost 계산식 

# Low level
# 첫번째 수식
print((y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean())

# 두번째 수식
print((y_one_hot * - F.log_softmax(z, dim=1)).sum(dim=1).mean())

# High level
# 세번째 수식
# nll이란 Negative Log Likelihood의 약자
# nll_loss는 F.log_softmax()를 수행한 후에 남은 수식들을 수행합니다. 
print(F.nll_loss(F.log_softmax(z, dim=1), y))

#이를 더 간단하게 하면 다음과 같이 사용할 수 있습니다. 
# F.cross_entropy()는 F.log_softmax()와 F.nll_loss()를 포함하고 있습니다.
# 네번째 수식
print(F.cross_entropy(z, y))