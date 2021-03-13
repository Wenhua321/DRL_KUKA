import torch
from kuka import KukaCamGymEnv, KukaCamGymEnv2, KukaCamGymEnv3
from agent import baseline, baseline2, baseline3, np_to_tensor, opt_cuda
import matplotlib.pyplot as plt
import numpy as np


def test_embedding(actor, mode='de'):
    def plot_hidden(h):
        for i in range(4):
            for j in range(4):
                plt.subplot(4, 4, i * 4 + j + 1)
                plt.imshow(h[i * 4 + j])
                plt.axis('off')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
        plt.show()
    env = KukaCamGymEnv(renders=False, mode=mode, width=actor.width)
    n_episodes = 1
    for n in range(n_episodes):
        s, info = env.reset()
        frame = 0
        R = 0
        while True:
            a = baseline(info)
            if frame == 40 or frame == 10:
                s_t = torch.tensor(s).type(torch.FloatTensor).unsqueeze(dim=0)
                with torch.no_grad():
                    if mode == 'rgbd':
                        h = actor.conv(s_t / 255)
                    else:
                        h1 = actor.conv(s_t[:, :3] / 255)
                        h2 = actor.conv(s_t[:, 3:] / 255)
                        h = torch.cat((h1, h2), dim=1)
                    plot_hidden(h.squeeze().numpy())
            s_next, r, done, info_next = env.step(a)
            s = s_next
            info = info_next
            R += r
            frame += 1
            if done or frame >= 100:
                print('episode', n+1, 'ends in', frame, 'frames, return =', R)
                break
    env.close()


def test_critic(critic, actor, baseline_ratio=1.0, render=False, n_episodes=1, mode='de', use_fast=True):
    env = KukaCamGymEnv2(renders=render, mode=mode, width=128, image_output=not use_fast)
    for n in range(n_episodes):
        s, info = env.reset()
        mark = 0
        frame = 0
        R = 0
        q_a_record = []
        q_b_record = []
        while True:
            a = baseline(info)
            if not use_fast:
                s_t = torch.tensor(s).type(torch.FloatTensor).unsqueeze(dim=0)
                info_t = torch.tensor(info[:8]).type(torch.FloatTensor).unsqueeze(dim=0)
            info = torch.tensor(info).type(torch.FloatTensor).unsqueeze(dim=0)
            with torch.no_grad():
                if not use_fast:
                    action = actor(s_t, info_t)
                else:
                    action = actor(info)
                q_a_record.append(critic(info, action).item())
                q_b_record.append(critic(info, torch.tensor(a).type(torch.FloatTensor).unsqueeze(dim=0)))
                action = action.squeeze().numpy()
            if np.random.uniform(0, 1) < baseline_ratio:
                s_next, r, done, info_next = env.step(a)
            else:
                s_next, r, done, info_next = env.step(action)
            s = s_next
            info = info_next
            R += r
            if r == 0.1:
                mark = frame
            frame += 1
            if done or frame >= 100:
                print('episode', n + 1, 'ends in', frame, 'frames, return =', R)
                plt.plot(q_a_record, label='agent')
                plt.plot(q_b_record, c='gray', alpha=0.5, label='baseline')
                if mark != 0:
                    plt.axvline(mark, c='red')
                plt.legend()
                plt.show()
                break
    env.close()


def test_actor(task, seed, log, n_episodes=100, render=False, mode='de', use_fast=True):
    with open(log + str(seed) + '/actor.pt', 'rb') as fa:
        actor = opt_cuda(torch.load(fa), 1)
    if task == 1:
        env = KukaCamGymEnv(renders=render, image_output=not use_fast, mode=mode, width=128)
    elif task == 2:
        env = KukaCamGymEnv2(renders=render, image_output=not use_fast, mode=mode, width=128)
    else:
        env = KukaCamGymEnv3(renders=render, image_output=not use_fast, mode=mode, width=128)
    success_count = 0
    sum_L = 0
    misbehavior_count = 0
    print("*******************************************")
    for n in range(n_episodes):
        s, info = env.reset()
        frame = 0
        R = 0
        while True:
            if not use_fast:
                s_t = np_to_tensor(s, 1).unsqueeze(dim=0)
                info_t = np_to_tensor(info[:8], 1).unsqueeze(dim=0)
            info = np_to_tensor(info, 1).unsqueeze(dim=0)
            with torch.no_grad():
                if not use_fast:
                    a = actor(s_t, info_t)
                else:
                    a = actor(info)
            a = a.cpu().squeeze().numpy()
            s_next, r, done, info_next = env.step(a)
            s = s_next
            info = info_next
            R += r
            frame += 1
            if done or frame >= 100:
                print('episode', n+1, 'ends in', frame, 'frames, return =', R)
                if done:
                    if R == 1:
                        sum_L += frame
                        success_count += 1
                    else:
                        misbehavior_count += 1
                break
    env.close()
    print('Average time in executing the task is', sum_L / success_count, ';\n'
          'Success rate in', n_episodes, 'episodes is', success_count / n_episodes, ';\n'
          'Misbehavior rate in', n_episodes, 'episodes is', misbehavior_count / n_episodes, ';\n')
    print("*******************************************")
    return sum_L / success_count, success_count / n_episodes, misbehavior_count / n_episodes


def test_baseline(n_episodes=100, render=False, add_noise=False):
    env = KukaCamGymEnv(renders=render, image_output=False)
    success_count = 0
    sum_L = 0
    misbehavior_count = 0
    print("*******************************************")
    for n in range(n_episodes):
        s, info = env.reset()
        frame = 0
        R = 0
        while True:
            a = baseline(info)
            if add_noise:
                a += 0.1 * np.random.normal(0, 1, 5)
            s_next, r, done, info_next = env.step(a)
            info = info_next
            frame += 1
            R += r
            if done or frame >= 100:
                print('episode', n+1, 'ends in', frame, 'frames, return =', R)
                if done:
                    if R == 1:
                        sum_L += frame
                        success_count += 1
                    else:
                        misbehavior_count += 1
                break
    env.close()
    print('Average time in executing the task is', sum_L / success_count, ';\n'
          'Success rate in', n_episodes, 'episodes is', success_count / n_episodes, ';\n'
          'Misbehavior rate in', n_episodes, 'episodes is', misbehavior_count / n_episodes, ';\n')
    print("*******************************************")


def test_baseline2(n_episodes=100, render=False, add_noise=False):
    env = KukaCamGymEnv2(renders=render, image_output=False)
    success_count = 0
    sum_L = 0
    print("*******************************************")
    for n in range(n_episodes):
        s, info = env.reset()
        frame = 0
        R = 0
        while True:
            a = baseline2(info)
            if add_noise:
                a += 0.1 * np.random.normal(0, 1, 5)
            s_next, r, done, info_next = env.step(a)
            info = info_next
            frame += 1
            R += r
            if done or frame >= 100:
                print('episode', n+1, 'ends in', frame, 'frames, return =', R)
                if done:
                    sum_L += frame
                    success_count += 1
                break
    env.close()
    print('Average time in executing the task is', sum_L / success_count, ';\n'
          'Success rate in', n_episodes, 'episodes is', success_count / n_episodes, ';\n')
    print("*******************************************")


def test_baseline3(n_episodes=100, render=False, add_noise=False):
    env = KukaCamGymEnv3(renders=render, image_output=False)
    success_count = 0
    sum_L = 0
    misbehavior_count = 0
    print("*******************************************")
    for n in range(n_episodes):
        s, info = env.reset()
        frame = 0
        R = 0
        while True:
            a = baseline3(info)
            if add_noise:
                a += 0.1 * np.random.normal(0, 1, 5)
            s_next, r, done, info_next = env.step(a)
            info = info_next
            frame += 1
            R += r
            if done or frame >= 100:
                print('episode', n+1, 'ends in', frame, 'frames, return =', R)
                if done:
                    if R == 1:
                        sum_L += frame
                        success_count += 1
                    else:
                        misbehavior_count += 1
                break
    env.close()
    print('Average time in executing the task is', sum_L / success_count, ';\n'
          'Success rate in', n_episodes, 'episodes is', success_count / n_episodes, ';\n'
          'Misbehavior rate in', n_episodes, 'episodes is', misbehavior_count / n_episodes, ';\n')
    print("*******************************************")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--task', type=int, default=1)
    # parser.add_argument('-q', '--mixed_q', action='store_true')
    # parser.add_argument('-b', '--baseline_boot', action='store_true')
    # parser.add_argument('-c', '--behavior_clone', action='store_true')
    # args = parser.parse_args()
    # exp = ['DDPG', 'wMQ', 'wBB', 'nBC', 'wBC', 'nBB', 'nMQ', '']
    # id = args.mixed_q + args.baseline_boot * 2 + args.behavior_clone * 4

    # with open('saves/t1/1/critic.pt', 'rb') as fc:
    #     critic = torch.load(fc)
    with open('saves/t1/1/actor.pt', 'rb') as fa:
        actor = torch.load(fa)

    test_embedding(actor, mode='de')
    # test_critic(critic, actor, render=False, baseline_ratio=1.0, n_episodes=1, mode='de')
    # test_critic(critic, actor, render=True, baseline_ratio=0.0, n_episodes=10, mode='de', use_fast=False)

    # log_file = 'saves/t' + str(args.task) + exp[id] + '/'
    # with open(log_file + 'actor_performance.csv', "w", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['av_frames', 'success_rate', 'misbehave_rate'])
    # for seed in range(1, 6):
    #     al, sr, mr = test_actor(task=args.task, seed=seed, log=log_file,
    #                             render=False, n_episodes=1000, mode='de', use_fast=True)
    #     with open(log_file + 'actor_performance.csv', "a+", newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(np.array([al, sr, mr]))

    # al, sr, mr = test_actor(task=1, seed=1, log='saves/t1/',
    #                         render=True, n_episodes=1000, mode='de', use_fast=False)
    # test_baseline(render=False, n_episodes=6000, add_noise=True)
    # test_baseline2(render=False, n_episodes=6000, add_noise=True)
    # test_baseline3(render=False, n_episodes=6000, add_noise=True)

