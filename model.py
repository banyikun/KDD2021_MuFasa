import random 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu" 
device = torch.device(dev)
print(device)


class Network(nn.Module):
    def __init__(self, dim, hidden_size=50):
        super(Network, self).__init__()
        self.fc1_t1 = nn.Linear(dim[0], hidden_size)
        self.fc2_t1 = nn.Linear(hidden_size, hidden_size)

        self.fc1_t2 = nn.Linear(dim[1], hidden_size)
        self.fc2_t2 = nn.Linear(hidden_size, hidden_size)
        
        self.fc3_com = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc4_com = nn.Linear(hidden_size*2, 1)
        
        self.activate = nn.ReLU()
    def forward(self, x1, x2):
        a = self.fc2_t1(self.activate(self.fc1_t1(x1)))
        b = self.fc2_t2(self.activate(self.fc1_t2(x2)))
        if len(a.shape)> 1:
            c = torch.cat((a,b), 1)
        else:
            c = torch.cat((a,b), 0)
        return self.fc4_com(self.activate(self.fc3_com(c)))
    

class MuFasa:
    def __init__(self, dim, num_tasks, lamdba=1, nu=0.1, hidden=100):
        self.func = Network(dim, hidden_size=hidden).to(device)
        #self.func_ini = Network(dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,)).to(device)
        self.nu = nu
        self.num_tasks = num_tasks
        self.dim = dim

    def select(self, context, t):
        tensor1 = torch.from_numpy(np.array(list(context[0]))).float().to(device)
        tensor2 = torch.from_numpy(np.array(list(context[1]))).float().to(device)
        
        def _factT(m):
            return np.sqrt(1 / (1 + m))
        
        mu = self.func(tensor1, tensor2)
        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        for fx in mu:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            sigma2 = self.lamdba * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            sample_r = fx.item() + self.nu *  sigma.item()
            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r
            g_list.append(g)
        arm = np.argmax(sampled)
        self.U += g_list[arm] * g_list[arm]
        return arm, g_list[arm].norm().item(), ave_sigma, ave_rew
    
    
    def update(self, context,  final_r, subrewards, t):
        self.context_list.append(context)
        self.reward.append(final_r)
        r1, r2 = subrewards[0], subrewards[1]
        new_cont1 = [context[0], np.zeros(self.dim[1])] 
        self.context_list.append(new_cont1)
        self.reward.append(r1)
        new_cont2 = [np.zeros(self.dim[0]), context[1]]
        self.context_list.append(new_cont2)
        self.reward.append(r2)

    def train(self):
        new_cont = self.context_list
        new_rwd =  self.reward
                
        optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba)
        length = len(new_rwd)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c1  = new_cont[idx][0]
                c2  = new_cont[idx][1]
                c1 = torch.from_numpy(c1).float().to(device)
                c2 = torch.from_numpy(c2).float().to(device)
                r = new_rwd[idx]
                optimizer.zero_grad()
                delta = self.func(c1,c2) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length


           

