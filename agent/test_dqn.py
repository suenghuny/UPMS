

from cfg import get_cfg
import os
import pandas as pd
from dqn import *
from agent.ppo import *
from environment.env import UPMSP

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    cfg = get_cfg()
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    num_episode = 15
    episode = 1
    score_avg = 0
    state_size =104
    action_size = 12
    lr = cfg.lr
    gamma = cfg.gamma
    lmbda = cfg.lmbda
    eps_clip = cfg.eps_clip
    K_epoch = cfg.K_epoch
    T_horizon = cfg.T_horizon

    w_delay = cfg.w_delay
    w_move = cfg.w_move
    w_priority = cfg.w_priority
    state_size = 104        # feature 1~8 size = 104 / 176
    action_size = 12
    log_path = '../result/model/dqn'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    event_path = '../environment/result/dqn'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    env = UPMSP(log_dir=event_path, num_j=1500,num_m=8, action_number = action_size, action_mode = 'WCOVERT')
    agent = Agent(state_size, action_size, lr, gamma, lmbda, eps_clip, K_epoch)

    agent.network.load_state_dict(torch.load('output/train/model/episode6900.pt')["model_state_dict"])

    # q = Qnet(state_size, action_size)
    # q.load_state_dict(torch.load(log_path + '/episode5000.pt')["model_state_dict"])       # 가중치가 저장되는 경로

    tard_list = list()
    #print(num_episode)
    for i in range(num_episode):
        env.e = i
        step = 0
        done = False
        s = env.reset()
        r = list()
        possible_actions = [True] * action_size
        while not done:
            epsilon = 0
            step += 1
            #action = 1
            a = 0
            #a, prob, mask = agent.get_action(s, possible_actions)

            # 환경과 연결
            next_state, reward, done = env.step(a)

            r.append(reward)
            s = next_state

            if done:
                env.monitor.save_tracer()
                break

        mean_wt = env.monitor.tardiness / env.num_job
        tard_list.append(mean_wt)
        print("Episode {0} | MWT = {1}".format(i, mean_wt))

    print("Total Mean Weighted Tardiness = ", np.mean(tard_list))