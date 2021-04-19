import numpy as np
import torch
from kuka import KukaCamGymEnv, KukaCamGymEnv2, KukaCamGymEnv3
from agent_1 import HLAgent
import csv
import argparse




def train(env, agent, log_dir, imitate=False):

    print('******************************************\n'
          'Start Training: \n'
          '     log_directory:', log_dir, ';\n'
          '******************************************')
    log_file = log_dir + '/log.csv'
    with open(log_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['episode', 'frames', 'return', 'misbehave', 'Lc', 'La', 'ratio', 'test'])
    if agent.use_fast:
        n_episodes = 3000
    else:
        n_episodes = 6000
    test = 0
    frames = 0
    for n in range(n_episodes):
        agent.actor.train()       
        s, info = env.reset()
        frame = 0
        R = 0
        ratio = 0
        if agent.use_fast and frames > 2e5:
            break
        elif frames > 4e5:
            break
        while True:
            action, flag = agent.act(s, info)
            if flag:
                ratio += 1
            if not imitate:
                action += 0.1 * np.random.normal(0, 1, 5)
            action = np.clip(action, -env.action_space.high, env.action_space.high)
            s_next, r, done, info_next = env.step(action)
            agent.remember(s, info, action, s_next, r, done, info_next)

            s = s_next
            info = info_next
            R += r
            frame += 1
            if done or frame >= 100:
                print('episode', n + 1, 'ends in', frame, 'frames, return', R, 'ratio', ratio / frame)
                frames += frame
                if not agent.use_fast:
                    for p in agent.optimizer_actor.param_groups:
                        p['lr'] = 1e-3 * (1 - 0.5 * frames / 4e5)
                av_Lc, av_La = agent.train()
                print('Lc:', av_Lc, 'La:', av_La)
                if (n + 1) % 30 == 0 and not imitate:
                    agent.actor.eval()
                
                    success_count = 0
                    if agent.use_fast:
                        test_episodes = 100
                    else:
                        test_episodes = 100
                    for _ in range(test_episodes):
                        s, info = env.reset()
                        frame_t = 0
                        R_t = 0
                        while True:
                            a, _ = agent.act(s, info, test=True)
                            s_next, r, done, info_next = env.step(a)
                            s = s_next
                            info = info_next
                            frame_t += 1
                            R_t += r
                            if done or frame_t >= 100:
                                if R_t == 1:
                                    success_count += 1
                                break
                    test = success_count / test_episodes
                new_line = np.array([n + 1, frames, R, frame < 100 and R == 0, av_Lc, av_La, ratio / frame, test])
                print(test)
                with open(log_file, "a+", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(new_line)
                with open(log_dir + '/critic.pt', 'wb') as fc:
                    torch.save(agent.critic, fc)

                with open(log_dir + '/actor.pt', 'wb') as fa:
                    torch.save(agent.actor, fa)
                break
    print('AGENT TERMINATED!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='de')
    parser.add_argument('-w', '--width', type=int, default=128)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-l', '--log', type=int, default=1)
    parser.add_argument('-i', '--use_image', action='store_true')
    parser.add_argument('-a', '--imitate', action='store_true')
    parser.add_argument('-e', '--delta',type =float,default=0.00005)
    parser.add_argument('-s', '--lamda',type =float,default=0.02)
    parser.add_argument('-t', '--task', type=int, default=3)
    parser.add_argument('-q', '--mixed_q', action='store_true')
    parser.add_argument('-b', '--baseline_boot', action='store_true')
    parser.add_argument('-c', '--behavior_clone', action='store_true')
    
    args = parser.parse_args()
    exp = ['DDPG', 'wMQ', 'wBB', 'nBC', 'wBC', 'nBB', 'nMQ', '']
    id = args.mixed_q + args.baseline_boot * 2 + args.behavior_clone * 4
    if args.task == 1:
        env = KukaCamGymEnv(renders=False, image_output=args.use_image, mode=args.mode, width=args.width)
    elif args.task == 2:
        env = KukaCamGymEnv2(renders=False, image_output=args.use_image, mode=args.mode, width=args.width)
    elif args.task == 3:
        env = KukaCamGymEnv3(renders=False, image_output=args.use_image, mode=args.mode, width=args.width)
    log_dir = 'save1/t' + str(args.task) + exp[id] + '/' + str(args.log)
    
    '''
    agent = HLAgent(mode=args.mode, width=args.width, device=args.gpu, use_fast=not args.use_image, task=args.task,
                    mixed_q=args.mixed_q, baseline_boot=args.baseline_boot,
                    behavior_clone=args.behavior_clone, imitate=args.imitate)
                  '''

    agent = HLAgent(mode=args.mode, width=args.width, device=args.gpu, use_fast=False, task=args.task,delta=args.delta,lamda=args.lamda,
                    mixed_q=args.mixed_q, baseline_boot=args.baseline_boot,
                    behavior_clone=args.behavior_clone, imitate=args.imitate)
    train(env, agent, log_dir=log_dir, imitate=args.imitate)
    
    '''
    agent = HLAgent(mode='de', width=128, device=0, use_fast=False, task=3,
                    mixed_q=True, baseline_boot=True,
                    behavior_clone=True, imitate=False)

    train(env, agent, log_dir=log_dir, imitate=False)
    '''
    
    env.close()
