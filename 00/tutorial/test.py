import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dqn import DQN, ReplayMemory


env = gym.make("LunarLander-v2", render_mode="human")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions, env).to(device)
target_net = DQN(n_observations, n_actions, env).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=policy_net.LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

episode_durations = []
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
def optimize_model():
    if len(memory) < policy_net.BATCH_SIZE:
        return
    
    transitions = memory.sample(policy_net.BATCH_SIZE)
    
    #샘플 뽑은거에서 Transition클래스로 들고와서 만듬
    batch = Transition(*zip(*transitions))

    # next_state의 상태가 None이 아닌 즉, 게임이 끝난 상태가 아닌경우에만 Q값을 없데이트함
    # lamda를 이용해 bach.next_state가 None이 아니면 True, None이면 False로 만들어서 반환한 텐서
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    
    # batch.next_state가 None이 아닌 것만 추출하여 (끝나지 않은 상태) 다음 상태 텐서를 만듬
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    # 하나의 텐서로 만드는 작업
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # policy_net(state_batch)를 이용하여 현재상태에서의 확률표를 뽑고 .gather를 이용하여 
    # 행동했던 텐서만 추출함
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # 다음 상태에 대한 Q값을 예측하기 위한 초기화 작업
    next_state_values = torch.zeros(policy_net.BATCH_SIZE, device=device)
    
    # 기울지 조정하지 않고
    with torch.no_grad():
        # DQN(Dueling Deep Q-Network) 알고리즘에서는 다음 상태에 대한 Q값을 계산할 때 
        # 해당 상태에서 취할 수 있는 모든 행동(action)에 대한 Q값 중 최댓값을 사용
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * policy_net.GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    print(f'{loss}')
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = policy_net.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*(policy_net.TAU) + target_net_state_dict[key]*(1-policy_net.TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()