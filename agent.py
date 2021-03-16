import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def xdist(pos, orn, x_n):
    return pos[0] + 0.02 * np.cos(orn[2] + np.pi / 2) - x_n


def ydist(pos, orn, y_n):
    return pos[1] + 0.02 * np.sin(orn[2] + np.pi / 2) - y_n


def da(angle, orn):
    return np.tanh(2 * (angle - orn))


def baseline(info):
    """
    :param info: (pos),(orn) of the gripper,
                 (finger angle, finger force),
                 (block_pos), (block_orn),
                 (cup_pos), (cup_orn)  21
    :return: action to go
    """
    x_n = info[0]
    y_n = info[1]
    height = info[2]
    gripper_angle = info[5]
    finger_angle = info[6]
    finger_force = info[7]
    block_pos = info[8:11]
    block_orn = info[11:14]
    cup_pos = info[14:17]
    cup_orn = info[17:20]
    action = np.zeros(5)
    if finger_force < 1:
        action[0] = np.tanh(xdist(block_pos, block_orn, x_n) * 5)
        action[1] = np.tanh(ydist(block_pos, block_orn, y_n) * 5)
        action[2] = np.tanh(5 * (block_pos[2] + 0.25 - height))
        action[3] = da(gripper_angle, block_orn[2])
        action[4] = da(0, finger_angle)
        if block_pos[2] + 0.25 < height < 0.35:
            action[4] = da(0.2, finger_angle)
    else:
        action[0] = np.tanh((cup_pos[0] - block_pos[0]) * 5)
        action[1] = np.tanh((cup_pos[1] - block_pos[1]) * 5)
        action[2] = np.tanh(5 * (0.25 - block_pos[2]))
        action[3] = da(gripper_angle, cup_orn[2])
        action[4] = da(0, finger_angle)
        if np.sqrt((cup_pos[0] - block_pos[0]) ** 2 + (cup_pos[1] - block_pos[1]) ** 2) < 0.02:
            action[4] = da(0.2, finger_angle)
    return action


def baseline2(info):
    x_n = info[0]
    y_n = info[1]
    height = info[2]
    gripper_angle = info[5]
    finger_angle = info[6]
    finger_force = info[7]
    block_pos = info[8:11]
    block_orn = info[11:14]
    block2_pos = info[14:17]
    block2_orn = info[17:20]
    action = np.zeros(5)
    if finger_force < 1:
        action[0] = np.tanh(xdist(block_pos, block_orn, x_n) * 5)
        action[1] = np.tanh(ydist(block_pos, block_orn, y_n) * 5)
        action[2] = np.tanh(5 * (block_pos[2] + 0.25 - height))
        action[3] = da(gripper_angle, block_orn[2])
        action[4] = da(0, finger_angle)
        if block_pos[2] + 0.25 < height < 0.35:
            action[4] = da(0.2, finger_angle)
    else:
        action[0] = np.tanh((block2_pos[0] - block_pos[0]) * 5)
        action[1] = np.tanh((block2_pos[1] - block_pos[1]) * 5)
        action[2] = np.tanh(5 * (0.1 - block_pos[2]))
        action[3] = da(gripper_angle, block2_orn[2])
        action[4] = da(0, finger_angle)
        if np.sqrt((block2_pos[0] - block_pos[0]) ** 2 + (block2_pos[1] - block_pos[1]) ** 2) < 0.02:
            action[4] = da(0.2, finger_angle)
    return action


def baseline3(info):
    x_n = info[0]
    y_n = info[1]
    height = info[2]
    gripper_angle = info[5]
    finger_angle = info[6]
    finger_force = info[7]
    cups_pos = info[8:11]
    cups_orn = info[11:14]
    cup_pos = info[14:17]
    cup_orn = info[17:20]
    action = np.zeros(5)
    if finger_force < 1:
        action[0] = np.tanh(xdist(cups_pos, cups_orn, x_n) * 5)
        action[1] = np.tanh(ydist(cups_pos, cups_orn, y_n) * 5)
        action[2] = np.tanh(5 * (cups_pos[2] + 0.3 - height))
        action[3] = da(gripper_angle, cups_orn[2])
        action[4] = da(0, finger_angle)
        if cups_pos[2] + 0.3 < height < 0.45:
            action[4] = da(0.2, finger_angle)
    else:
        action[0] = np.tanh((cup_pos[0] - cups_pos[0]) * 5)
        action[1] = np.tanh((cup_pos[1] - cups_pos[1]) * 5)
        action[2] = np.tanh(5 * (0.35 - cups_pos[2]))
        action[3] = da(gripper_angle, cup_orn[2])
        action[4] = da(0, finger_angle)
        if np.sqrt((cup_pos[0] - cups_pos[0]) ** 2 + (cup_pos[1] - cups_pos[1]) ** 2) < 0.02:
            action[4] = da(0.2, finger_angle)
    return action


def baseline_controller(info, base):
    size = info.shape[0]
    action = np.zeros((size, 5), dtype=np.float32)
    for i in range(size):
        action[i, :] = base(info[i, :])
    return action


def opt_cuda(t, device):
    if torch.cuda.is_available():
        cuda = "cuda:" + str(device)
        return t.cuda(cuda)
    else:
        return t


def np_to_tensor(n, device):
    return opt_cuda(torch.from_numpy(n).type(torch.FloatTensor), device)


def soft_update(target, source, tau):
    # 网络参数复制   source->target, tau:比例
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)




'''
class Actor(nn.Module):
    def __init__(self, mode, width, device):
        super(Actor, self).__init__()
        self.mode = mode
        self.width = width
        self.device = device
        self.in_channel = 4 if self.mode == 'rgbd' else 3
        self.out_channel = 16 if self.mode == 'rgbd' else 8
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, 3, 1),
            nn.ReLU(),
            nn.Conv2d(self.out_channel, self.out_channel, 3, 1),
            nn.ReLU(),
            nn.Conv2d(self.out_channel, self.out_channel, 3, 1),
            nn.ReLU())

        # 4*128*128->16*122*122
        self.fc1 = nn.Sequential(
            nn.Linear(32 + 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
            nn.Tanh())




    def forward(self, x, robot_state):
        if self.mode == 'rgbd':
            x2 = self.conv(x / 255)  # 122*122*8
        elif self.mode == 'de':
            l1 = self.conv(x[:, :3] / 255)
            l2 = self.conv(x[:, 3:] / 255)
            x2 = torch.cat((l1, l2), dim=1)
        x3 = SpacialSoftmaxExpectation(self.width - 6, self.device)(x2)
        # concatenate with robot state:
        action = self.fc1(torch.cat((x3, robot_state), dim=1))  # 32 + 8
        return action


'''
class SpacialSoftmaxExpectation(nn.Module):
    def __init__(self, size, device):
        super(SpacialSoftmaxExpectation, self).__init__()
        cor = opt_cuda(torch.arange(size).type(torch.FloatTensor), device)
        X, Y = torch.meshgrid(cor, cor)
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        self.fixed_weight = torch.cat((Y, X), dim=1)
        self.fixed_weight /= size - 1

    def forward(self, x):
        return nn.Softmax(2)(x.view(*x.size()[:2], -1)).matmul(self.fixed_weight).view(x.size(0), -1)


class Actor(nn.Module):
    def __init__(self, mode, width, device):
        super(Actor, self).__init__()
        self.mode = mode
        self.width = width
        self.device = device
        self.in_channel = 4 if self.mode == 'rgbd' else 3
        self.out_channel = 16 if self.mode == 'rgbd' else 8
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, 3, 1),
            nn.ReLU(),
            nn.Conv2d(self.out_channel, self.out_channel, 3, 1),
            nn.ReLU(),
            nn.Conv2d(self.out_channel, self.out_channel, 3, 1),
            nn.ReLU())

        # 4*128*128->16*122*122
        self.fc1 = nn.Sequential(
            nn.Linear((32 + 8)*5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
            nn.Tanh())




    def forward(self, x, robot_state):
        output = torch.tensor(40 * 5, torch.float, x.device())
        for i in range(5):
            if self.mode == 'rgbd':
                x2 = self.conv(x[i] / 255)  # 122*122*8
            elif self.mode == 'de':
                l1 = self.conv(x[i,:, :3] / 255)
                l2 = self.conv(x[i,:, 3:] / 255)
                x2 = torch.cat((l1, l2), dim=1)
            x3 = SpacialSoftmaxExpectation(self.width - 6, self.device)(x2)
            # concatenate with robot state:
            output[i * 40:(i + 1) * 128] = torch.cat((x3, robot_state[i]), dim=1)
        action = self.fc1(output)  # 32 + 8
        return action


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(20 + 5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, info, action):
        q = self.fc1(torch.cat((info, action), dim=1))
        return q

class FastActor(nn.Module):
    def __init__(self):
        super(FastActor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(20, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
            nn.Tanh())

    def forward(self, info):
        a = self.fc1(info)
        return a

'''
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(20 + 5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, info, action):
        q = self.fc1(torch.cat((info, action), dim=1))
        return q
   
class ReplayBuffer:
    def __init__(self,mode, w, h, action_dim, info_dim, size):

        c = 6 if mode == 'de' else 4
        self.sta1_buf = np.zeros([size, c, h, w], dtype=np.uint8)
        self.sta2_buf = np.zeros([size, c, h, w], dtype=np.uint8)
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.done_buf = np.zeros([size, 1], dtype=np.bool_)
        self.info1_buf = np.zeros([size, info_dim], dtype=np.float32)
        self.info2_buf = np.zeros([size, info_dim], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, sta, info, act, next_sta, rew, done, next_info):
        self.sta1_buf[self.ptr] = sta
        self.sta2_buf[self.ptr] = next_sta
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.info1_buf[self.ptr] = info
        self.info2_buf[self.ptr] = next_info
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):

        idxs = np.random.randint(0, self.size, size=batch_size)

        return dict(sta1=self.sta1_buf[idxs],
                    sta2=self.sta2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    info1=self.info1_buf[idxs],
                    info2=self.info2_buf[idxs])
'''

class ReplayBuffer:
    def __init__(self,length,mode, w, h, action_dim, info_dim, size):

        c = 6 if mode == 'de' else 4
        self.sta1_buf = np.zeros([size, length, c, h, w], dtype=np.uint8)
        self.sta2_buf = np.zeros([size, length, c, h, w], dtype=np.uint8)
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.done_buf = np.zeros([size, 1], dtype=np.bool_)
        self.info1_buf = np.zeros([size, length,info_dim], dtype=np.float32)
        self.info2_buf = np.zeros([size,length, info_dim], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, sta, info, act, next_sta, rew, done, next_info):
        self.sta1_buf[self.ptr] = sta
        self.sta2_buf[self.ptr] = next_sta
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.info1_buf[self.ptr] = info
        self.info2_buf[self.ptr] = next_info
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):

        idxs = np.random.randint(0, self.size, size=batch_size)

        return dict(sta1=self.sta1_buf[idxs],
                    sta2=self.sta2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    info1=self.info1_buf[idxs],
                    info2=self.info2_buf[idxs])


class Observation():
    def __init__(self,length,c,h,w,info_dim):
        self.sta = np.zeros([length, c, h, w], dtype=np.uint8)
        self.info = np.zeros([length, info_dim], dtype=np.float32)
        self.ptr,  self.max_size = 0, length
    def store(self,observation,info):
        if self.ptr == 0:
            for i in range(self.max_size):
                self.sta[i] = observation
                self.info[i] = info
            self.ptr = 1
        else:
            for i in range(self.max_size-1):
                self.sta[i] = self.sta[i+1]
                self.info[i] = self.info[i+1]
            self.sta[self.max_size-1] = observation
            self.info[self.max_size-1] = info
    def clean(self):
        self.ptr = 0



class ReplayBufferFast:
    def __init__(self, action_dim, info_dim, size):
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.done_buf = np.zeros([size, 1], dtype=np.bool_)
        self.info1_buf = np.zeros([size, info_dim], dtype=np.float32)
        self.info2_buf = np.zeros([size, info_dim], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, info, act, rew, done, next_info):
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.info1_buf[self.ptr] = info
        self.info2_buf[self.ptr] = next_info
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    info1=self.info1_buf[idxs],
                    info2=self.info2_buf[idxs])


class HLAgent:
    def __init__(self, mode='de', width=128, device=0, use_fast=False, task=1,
                 mixed_q=True, baseline_boot=True, behavior_clone=True, imitate=False):
        self.mode = mode
        self.width = width
        self.device = device
        self.use_fast = use_fast
        self.mixed_q = mixed_q
        self.baseline_boot = baseline_boot
        self.behavior_clone = behavior_clone
        self.imitate = imitate
        self.length = 5
        if imitate:
            with open('best/'+str(task)+'/actor.pt', 'rb') as fa:
                self.best = opt_cuda(torch.load(fa), self.device)
        if task == 1:
            self.base = baseline
        elif task == 2:
            self.base = baseline2
        elif task == 3:
            self.base = baseline3
        if self.use_fast:
            self.buffer = ReplayBufferFast(5, 20, size=100000)
        else:
            self.buffer = ReplayBuffer(self.length,self.mode, self.width, self.width, 5, 20, size=100000)
        print('******************************************\n'
              'Create Agent: \n'
              '     task:', task, ';\n'
              '     mode:', self.mode, ';\n'
              '     image_width:', self.width, ';\n'
              '     device:', self.device, ';\n'
              '     imitate?:', self.imitate, ';\n'
              '     use fast actor?:', self.use_fast, ';\n'
              '     use mixed Q control?:', self.mixed_q, ';\n'
              '     use baseline bootstrap?:', self.baseline_boot, ';\n'
              '     use behavior cloning?:', self.behavior_clone, ';\n'
              '******************************************')
        if self.use_fast:
            self.actor = opt_cuda(FastActor(), self.device)
            self.target_actor = opt_cuda(FastActor(), self.device)
        else:
            self.actor = opt_cuda(Actor(mode=self.mode, width=self.width, device=self.device), self.device)
            self.target_actor = opt_cuda(Actor(mode=self.mode, width=self.width, device=self.device), self.device)
        self.critic = opt_cuda(Critic(), self.device)
        self.target_critic = opt_cuda(Critic(), self.device)
        soft_update(self.target_actor, self.actor, 1)
        soft_update(self.target_critic, self.critic, 1)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.tau = 0.005
        self.epsilon = 1
        self.delta = 2e-5
        self.batch_size = 256
        self.train_step = 50

    def act(self, s, info, test=False):
        action_b = self.base(info[4])
        info = np_to_tensor(info, self.device).unsqueeze(dim=0)
        if self.use_fast:
            with torch.no_grad():
                action = self.actor(info)
        else:
            s = np_to_tensor(s, self.device).unsqueeze(dim=0)
            info_t = info[:,:, :8]
            with torch.no_grad():
                action = self.actor(s, info_t)
        if test or self.imitate:
            return action.squeeze().cpu().numpy(), True
        if np.random.uniform(0, 1) < self.epsilon:
            self.epsilon = max(self.epsilon - self.delta, 0)
            return action_b, False
        else:
            self.epsilon = max(self.epsilon - self.delta, 0)
            action_b_t = np_to_tensor(action_b, self.device).unsqueeze(dim=0)
            with torch.no_grad():
                q_b = self.critic(info[4], action_b_t)
                q = self.critic(info[4], action)
            if q_b.item() > q.item() and self.mixed_q:
                return action_b, False
            else:
                return action.squeeze().cpu().numpy(), True

    def remember(self, state, info, action, next_state, reward, done, next_info):
        if self.use_fast:
            self.buffer.store(info, action, [reward], [done], next_info)
        else:
            self.buffer.store(state, info, action, next_state, [reward], [done], next_info)

    def train(self):
        total_Lc = total_La = 0
        # steps = min(self.train_step, self.buffer.size // self.batch_size + 1)
        steps = self.train_step
        for i in range(steps):
            batch = self.buffer.sample_batch(batch_size=self.batch_size)
            if not self.use_fast:
                si = np_to_tensor(batch['sta1'], self.device).contiguous()
                sn = np_to_tensor(batch['sta2'], self.device).contiguous()
                infoi_t = np_to_tensor(batch['info1'][:,:, :8], self.device)
                infon_t = np_to_tensor(batch['info2'][:,:, :8], self.device)
            ai = np_to_tensor(batch['acts'], self.device)
            ri = np_to_tensor(batch['rews'], self.device)
            d = np_to_tensor(batch['done'], self.device)
            infoi = np_to_tensor(batch['info1'], self.device)

            infon = np_to_tensor(batch['info2'], self.device)
            baseline_action = np_to_tensor(baseline_controller(batch['info1'][:,4,:], self.base), self.device)
            baseline_action_n = np_to_tensor(baseline_controller(batch['info2'][:,4,:], self.base), self.device)

            if self.imitate:
                self.optimizer_actor.zero_grad()
                if self.use_fast:
                    a = self.actor(infoi)
                else:
                    a = self.actor(si, infoi_t)
                with torch.no_grad():
                    best_action = self.best(infoi)
                La = ((a - best_action) ** 2).mean()
                La.backward()
                self.optimizer_actor.step()
                total_La += La.item()

            else:
                self.optimizer_critic.zero_grad()
                with torch.no_grad():
                    if self.use_fast:
                        back_up = self.target_critic(infon, self.target_actor(infon))
                    else:

                        back_up = self.target_critic(infon[:,4], self.target_actor(sn, infon_t))
                    if self.baseline_boot:
                        back_up_d = self.target_critic(infon[:,4], baseline_action_n)
                        back_up = torch.max(back_up, back_up_d)
                    yi = ri + (1 - d) * self.gamma * back_up
                Lc = ((yi - self.critic(infoi[:,4], ai)) ** 2).mean()
                Lc.backward()
                self.optimizer_critic.step()

                self.optimizer_actor.zero_grad()
                if self.use_fast:
                    a = self.actor(infoi)
                else:
                    a = self.actor(si, infoi_t)
                q_a = self.critic(infoi[:,4], a)
                if self.behavior_clone:
                    with torch.no_grad():
                        q_a_d = self.critic(infoi[:,4], baseline_action)
                        xi = nn.ReLU()(torch.sign(q_a_d - q_a)).contiguous()
                    La = (((a - baseline_action) ** 2).mean() * xi - 0.02 * q_a).mean().contiguous()
                else:
                    La = - 0.02 * q_a.mean()
                La.backward()
                self.optimizer_actor.step()

                soft_update(self.target_critic, self.critic, self.tau)
                soft_update(self.target_actor, self.actor, self.tau)

                total_Lc += Lc.item()
                total_La += La.item()

        return total_Lc / steps, total_La / steps