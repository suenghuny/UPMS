import csv
import os
from dqn import *
from environment.env import UPMSP
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    num_episode = 10000
    episode = 1
    state_size = 104    # feature 1~8 size = 104 / 176
    action_size = 20
    log_path = '../result/model/dqn'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    event_path = '../environment/result/dqn'
    if not os.path.exists(event_path):
        os.makedirs(event_path)



    env = UPMSP(log_dir=event_path, num_j=200,num_m=8, action_number = action_size)         # 환경을 설정하기 위한 파라미터
    q = Qnet(state_size, action_size).to(device)
    q_target = Qnet(state_size, action_size).to(device)
    optimizer = optim.Adam(q.parameters(), lr=1e-5, eps=1e-06) # learning rate 변경
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    update_interval = 20
    save_interval = 100
    score = 0
    step = 0
    moving_average = list()
    cumulative_rewards = list()


    for e in range(episode, episode + num_episode + 1):
        import time

        start = time.time()
        done = False
        step = 0
        state = env.reset()
        r = list()
        loss = 0
        num_update = 0

        while not done:
            epsilon = max(0.01, 0.1 - 0.01 * (e / 200))
            step += 1
            action = q.sample_action(torch.from_numpy(state).float().to(device), epsilon)
            # 환경과 연결
            next_state, reward, done = env.step(action)
            r.append(reward)
            memory.put((state, action, reward, next_state, done))

            if memory.size() > 2000:
                loss += train(q, q_target, memory, optimizer)
                num_update += 1

            state = next_state
            if e % update_interval == 0 and e != 0:
                q_target.load_state_dict(q.state_dict())
            if done:
                writer.add_scalar("episode_reward/train", np.sum(r), e)
                if e % save_interval == 0 and e > 0:
                    torch.save({'episode': e,
                                'model_state_dict': q_target.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               log_path + '/episode%d.pt' % (e))
                    print('save model...')
                break

        cumulative_rewards.append(np.sum(r))
        print(e, np.sum(r), time.time() - start)


            # if e % 1000 == 0 and e > 0:  # 전시
            #     moving_average_rewards = list()
            #     for k in range(len(cumulative_rewards) - 20):
            #         moving_average_rewards.append(np.mean(cumulative_rewards[k:k + 20]))
            #     plt.plot(cumulative_rewards, label='cumulative_rewards')
            #     plt.plot(moving_average_rewards, label='moving average')
            #     plt.legend()
            #     plt.show()


