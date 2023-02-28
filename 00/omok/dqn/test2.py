import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from Omok4 import Omok
from actor import ActorCritic
from rule import *
from pygame.locals import *


env = Omok(10)

# 텐서 계산에 GPU사용하기위해 device에 gpu변수 세팅
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 액션수와 state 수 세팅
n_actions = len(env.get_env())
n_observations = len(env.get_env())

print(n_observations)

# 플레이어1,2 에이전트 생성
model = ActorCritic(device).to(device)
model2 = ActorCritic(device).to(device)

steps_done = 0
score = 0.0
score2 = 0.0
print_interval = 20

if torch.cuda.is_available():
    num_episodes = 1000000
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    episode_end = True
    env.reset()
    s = env.get_env()
    s = torch.tensor(s, dtype=torch.float32, device=device)
    
    while episode_end:
        for t in range(model.n_rollout):
            prob = model.pi(s)
            
            # 여기부터 s에서 바둘돌이 두여진 (0이 아닌곳은 action에서 제외하고 뽑음)
            board_tensor = torch.tensor(s, device=device, dtype=torch.long)
            mask = (board_tensor == 0)
            x_flatten = mask.flatten()
            true_indices = torch.nonzero(x_flatten).flatten()
            board_tensor = prob.flatten()
            selected_values = torch.gather(board_tensor, 0, true_indices)
            indicate = torch.nonzero(board_tensor == selected_values.max(0)[0])
            ####################################################################
            a = torch.tensor([[indicate]], device=device, dtype=torch.long)
            
            # m = Categorical(prob)
            # a = m.sample().item()
            s_prime, r, done = env.step(a, 1)
            episode_end = not done
            s_prime = torch.tensor(s_prime, dtype=torch.float32, device=device)

            if done:
                score += r
                model.put_data((s, a, r, s_prime, done))
                break
            
            prob2 = model2.pi(s_prime)
            mask2 = (s_prime == 0)
            x_flatten2 = mask2.flatten()
            true_indices2 = torch.nonzero(x_flatten2).flatten()
            board_tensor2 = prob2.flatten()
            selected_values2 = torch.gather(board_tensor2, 0, true_indices2)
            indicate2 = torch.nonzero(board_tensor2 == selected_values2.max(0)[0])
                
            a2 = torch.tensor([[indicate2]], device=device, dtype=torch.long)
            
            s_prime2, r2, done2 = env.step(a2, 2)
            episode_end = not done2
            s_prime2 = torch.tensor(s_prime2, dtype=torch.float32, device=device)
            
            model.put_data((s, a, r, s_prime2, done))
            model2.put_data((s_prime, a2, r2, s_prime2, done2))
            
            s = s_prime2
            score += r
            score2 += r2
            if done2:
                break
                  
        model.train_net()
            
    if i_episode%print_interval==0 and i_episode!=0:
            print("# of episode :{}, avg reward1 : {:.1f}".format(i_episode, score/print_interval))
            print("# of episode :{}, avg reward2 : {:.1f}".format(i_episode, score2/print_interval))
            score = 0
        
print('Complete')
# plot_durations(show_result=True)
# plt.ioff()
# plt.show()
