import numpy as np
import torch
import torch.nn as nn


def opt_cuda(t):
    if torch.cuda.is_available():
        cuda = "cuda:0"
        return t.cuda(cuda)
    else:
        return t





class SpacialSoftmaxExpectation(nn.Module):
    def __init__(self, size):
        super(SpacialSoftmaxExpectation, self).__init__()
        cor = opt_cuda(torch.arange(size).type(torch.FloatTensor))
        X, Y = torch.meshgrid(cor, cor)
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        self.fixed_weight = torch.cat((Y, X), dim=1)
        self.fixed_weight /= size - 1

    def forward(self, x):
        return nn.Softmax(2)(x.view(*x.size()[:2], -1)).matmul(self.fixed_weight).view(x.size(0), -1)


class Embedding(nn.Module):
    def __init__(self, mode, width):
        super(Embedding, self).__init__()
        self.mode = mode
        self.width = width
        self.in_channel = 4 if self.mode == 'rgbd' else 3
        self.out_channel = 64 if self.mode == 'rgbd' else 32
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, int(self.out_channel/4), 3, 1),
            nn.ReLU(),
            nn.Conv2d(int(self.out_channel/4), int(self.out_channel/2), 3, 1),
            nn.ReLU(),
            nn.Conv2d(int(self.out_channel/2), self.out_channel, 3, 1),
            nn.ReLU())

        self.fc1 = nn.Sequential(
            nn.Linear(128 + 8, 128),
            nn.ReLU())

    def forward(self, x, robot_state):

        output = torch.tensor(128*5,torch.float,x.device())
        for i in range(5):
            if self.mode == 'rgbd':
                x2 = self.conv(x[i] / 255)
            else:
                l1 = self.conv(x[i,:, :3] / 255)
                l2 = self.conv(x[i,:, 3:] / 255)
                x2 = torch.cat((l1, l2), dim=1)
            x3 = SpacialSoftmaxExpectation(self.width - 6)(x2)
        # concatenate with robot state:
            output[i*128:(i+1)*128] = self.fc1(torch.cat((x3, robot_state[i]), dim=1))

        return output



class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(128*5, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
            nn.Tanh())

    def forward(self, latent):
        a = self.fc1(latent)
        return a


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(20 + 5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh())

    def forward(self, info, action):
        q = self.fc1(torch.cat((info, action), dim=1))
        return q


class ReplayBuffer:
    def __init__(self,frames,c, w, h, action_dim, info_dim, size):
        self.sta1_buf = np.zeros([size,frames, c, h, w], dtype=np.uint8)
        self.sta2_buf = np.zeros([size,frames, c, h, w], dtype=np.uint8)
        self.acts_buf = np.zeros([size,action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.done_buf = np.zeros([size, 1], dtype=np.bool_)
        self.info1_buf = np.zeros([size,frames, info_dim], dtype=np.float32)
        self.info2_buf = np.zeros([size,frames, info_dim], dtype=np.float32)
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

class ObservationWithMemory():
    def __init__(self,size,c,h,w,info_dim):
        self.sta_buf = np.zeros([size, c, h, w], dtype=np.uint8)
        self.info_buf = np.zeros([size, info_dim], dtype=np.float32)
        self.ptr,  self.max_size = 0, size
    def store(self,observation,info):
        if self.ptr == 0:
            for i in range(self.max_size):
                self.sta_buf[i] = observation
                self.info_buf[i] = info
            self.ptr = 1
        else:
            for i in range(self.max_size-1):
                self.sta_buf[i] = self.sta_buf[i+1]
                self.info_buf[i] = self.info_buf[i+1]
            self.sta_buf[self.max_size-1] = observation
            self.info_buf[self.max_size-1] = info
    def clean(self):
        self.ptr = 0



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
                 (cup_pos), (cup_orn)
    :return: action to go
    BENCHMARK 1000: 0.954-0.925
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
        action[0] = np.tanh(xdist(block_pos, block_orn, x_n) * 6 / (1.55 - height) ** 2)
        action[1] = np.tanh(ydist(block_pos, block_orn, y_n) * 6 / (1.55 - height) ** 2)
        action[2] = np.tanh(5 * (block_pos[2] + 0.21 - height))
        action[3] = da(gripper_angle, block_orn[2])
        action[4] = da(0, finger_angle)
        if block_pos[2] + 0.24 < height < 0.36:
            action[4] = da(0.2, finger_angle)
    else:
        action[0] = np.tanh((cup_pos[0] - block_pos[0]) * 15 / (1.3 - block_pos[2]) ** 6)
        action[1] = np.tanh((cup_pos[1] - block_pos[1]) * 15 / (1.3 - block_pos[2]) ** 6)
        action[2] = np.tanh(15 * (0.2 - block_pos[2]))
        action[3] = da(gripper_angle, cup_orn[2])
        action[4] = da(0, finger_angle)
        if np.sqrt((cup_pos[0] - block_pos[0]) ** 2 + (cup_pos[1] - block_pos[1]) ** 2) < 0.015:
            action[4] = da(2, finger_angle)
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


def baseline_controller(info):
    """
    :param info: batch sized info: np array, size*20
    :return: action: batch sized action: np array, size*5
    """
    size = info.shape[0]
    action = np.zeros((size, 5), dtype=np.float32)
    for i in range(size):
        action[i, :] = baseline(info[i, :])
    return action


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class HLAgent:
    def __init__(self, nobuffer=False, recycle=False, epsilon=1.0, mode='rgbd', width=100):
        self.nobuffer = nobuffer
        self.mode = mode
        self.width = width

        if not self.nobuffer:
            self.buffer = ReplayBuffer(5,4 if self.mode == 'rgbd' else 6, self.width, self.width, 5, 20,
                                       size=int(100000 / ((self.width ** 2) / 1e4)) if self.mode == 'rgbd'
                                       else int(7000 / ((self.width ** 2) / 1e4)))
        else:
            self.buffer = ReplayBuffer(4 if self.mode == 'rgbd' else 6, self.width, self.width, 5, 20, size=1)
        if recycle:
            if torch.cuda.is_available():
                with open('temp/embedding.pt', 'rb') as fe:
                    self.embedding = opt_cuda(torch.load(fe))
                with open('temp/actor.pt', 'rb') as fa:
                    self.actor = opt_cuda(torch.load(fa))
                with open('temp/critic.pt', 'rb') as fc:
                    self.critic = opt_cuda(torch.load(fc))
            else:
                with open('temp/embedding.pt', 'rb') as fe:
                    self.embedding = opt_cuda(torch.load(fe, map_location=torch.device('cpu')))
                with open('temp/actor.pt', 'rb') as fa:
                    self.actor = opt_cuda(torch.load(fa, map_location=torch.device('cpu')))
                with open('temp/critic.pt', 'rb') as fc:
                    self.critic = opt_cuda(torch.load(fc, map_location=torch.device('cpu')))
        else:
            self.embedding = opt_cuda(Embedding(mode=self.mode, width=self.width))
            self.actor = opt_cuda(Actor())
            self.critic = opt_cuda(Critic())
        self.target_embedding = opt_cuda(Embedding(mode=self.mode, width=self.width))
        self.target_actor = opt_cuda(Actor())
        self.target_critic = opt_cuda(Critic())
        soft_update(self.target_embedding, self.embedding, 1)
        soft_update(self.target_actor, self.actor, 1)
        soft_update(self.target_critic, self.critic, 1)
        self.optimizer = torch.optim.Adam([{'params': self.embedding.parameters()},
                                           {'params': self.actor.parameters()},
                                           {'params': self.critic.parameters()}], lr=2e-3)
        self.loss = nn.MSELoss()
        self.gamma = 0.99
        self.tau = 0.01
        self.epsilon = epsilon
        self.batch_size = 128
        self.train_step = 30
        self.life_length = 0

    def act(self, s, info, test=False):
        if np.random.uniform(0, 1) > self.epsilon or test:
            s = opt_cuda(torch.from_numpy(s).type(torch.FloatTensor).unsqueeze(dim=0))
            info = opt_cuda(torch.from_numpy(info[:,:8]).type(torch.FloatTensor).unsqueeze(dim=0))
            with torch.no_grad():
                latent = self.embedding(s, info)
                action = self.actor(latent)
            action = action.squeeze().cpu().numpy()
        else:
            action = baseline_controller(info[4].reshape(1, -1)).squeeze()
        self.life_length += 1
        return action

    def remember(self, state, info, action, next_state, reward, done, next_info):
        if self.nobuffer:
            return
        self.buffer.store(state, info, action, next_state, [reward], [done], next_info)

    def train(self, show_log=False):
        total_Lc = 0
        total_La = 0
        if self.nobuffer:
            return 0, 0
        step = min(self.train_step, int(self.buffer.size / self.batch_size) + 1)
        for p in self.optimizer.param_groups:
            p['lr'] = self.epsilon * 2e-3
        for i in range(step):
            batch = self.buffer.sample_batch(batch_size=self.batch_size)
            si = batch['sta1']
            infoi = batch['info1']
            ai = batch['acts']
            sn = batch['sta2']
            ri = batch['rews']
            d = batch['done']
            infon = batch['info2']

            si = opt_cuda(torch.from_numpy(si).type(torch.FloatTensor))
            ai = opt_cuda(torch.from_numpy(ai).type(torch.FloatTensor))
            sn = opt_cuda(torch.from_numpy(sn).type(torch.FloatTensor))
            ri = opt_cuda(torch.from_numpy(ri).type(torch.FloatTensor))
            d = opt_cuda(torch.from_numpy(d).type(torch.FloatTensor))
            infoi_t = opt_cuda(torch.from_numpy(infoi[:, :8]).type(torch.FloatTensor))
            infon_t = opt_cuda(torch.from_numpy(infon[:, :8]).type(torch.FloatTensor))
            baseline_action = opt_cuda(torch.from_numpy(baseline_controller(infoi)).type(torch.FloatTensor))
            infoi = opt_cuda(torch.from_numpy(infoi).type(torch.FloatTensor))
            infon = opt_cuda(torch.from_numpy(infon).type(torch.FloatTensor))

            self.optimizer.zero_grad()
            latenti = self.embedding(si, infoi_t)
            a = self.actor(latenti)
            q_a = self.critic(infoi[4], a)
            with torch.no_grad():
                latentn = self.target_embedding(sn, infon_t)
                yi = ri + (1 - d) * self.gamma * self.target_critic(infon[4], self.target_actor(latentn))
                q_b = self.critic(infoi[4], baseline_action)
                xi = nn.ReLU()(q_b - q_a)
            La = (((baseline_action - a) ** 2).mean(dim=1) * xi * 2e4 - q_a).mean()
            La.backward(retain_graph=True)
            self.critic.zero_grad()
            Lc = self.loss(yi, self.critic(infoi[4], ai)) * 1e3
            Lc.backward(retain_graph=True)
            self.optimizer.step()

            soft_update(self.target_embedding, self.embedding, self.tau)
            soft_update(self.target_critic, self.critic, self.tau)
            soft_update(self.target_actor, self.actor, self.tau)

            total_Lc += Lc.item()
            total_La += La.item()

            if show_log and ((i + 1) % 10 == 0):
                print('step', i + 1, ': Lc', Lc.item(), 'La', La.item())

        return total_Lc / step, total_La / step
