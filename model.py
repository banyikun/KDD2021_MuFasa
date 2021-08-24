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
        self.fc1_t2 = nn.Linear(dim[1], hidden_size)
        self.fc2_t1 = nn.Linear(hidden_size, hidden_size)
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

    def train(self, context,  final_r, subrewards, t):
       
        self.context_list.append(context)
        self.reward.append(final_r)
        r1, r2 = subrewards[0], subrewards[1]
        new_cont1 = [context[0], np.zeros(self.dim[1])] 
        self.context_list.append(new_cont1)
        self.reward.append(r1)
        new_cont2 = [np.zeros(self.dim[0]), context[1]]
        self.context_list.append(new_cont2)
        self.reward.append(r2)
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


            
           

class Network_1(nn.Module):
    def __init__(self, dim, hidden_size=50):
        super(Network_1, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.ac = nn.ReLU()
    def forward(self, x):
        mid = self.fc2(self.ac(self.fc1(x)))   
        return self.fc3(self.ac(mid))
    
    
    
class Network_com(nn.Module):
    def __init__(self, dim, hidden_size=50):
        super(Network_com, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.ac = nn.ReLU()
    def forward(self, x):
        mid = self.fc2(self.ac(self.fc1(x)))
        return self.fc3(self.ac(mid))
    

    

class MuFasa_2:
    def __init__(self, dim, num_tasks, lamdba=1, nu=0.1, hidden=100):
        self.func_1 = Network_1(dim[0]).to(device)
        self.func_2 = Network_1(dim[1]).to(device)
        self.func_3 = Network_1(2).to(device)
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param_1 = sum(p.numel() for p in self.func_1.parameters() if p.requires_grad)
        self.U_1 = lamdba * torch.ones((self.total_param_1,)).to(device)
        self.total_param_2 = sum(p.numel() for p in self.func_2.parameters() if p.requires_grad)
        self.U_2 = lamdba * torch.ones((self.total_param_2,)).to(device)
        self.total_param_3 = sum(p.numel() for p in self.func_3.parameters() if p.requires_grad)
        self.U_3 = lamdba * torch.ones((self.total_param_3,)).to(device)
        self.nu = nu
        self.num_tasks = num_tasks
        self.dim = dim

    def select(self, context, t):
        tensor1 = torch.from_numpy(np.array(list(context[0]))).float().to(device)
        tensor2 = torch.from_numpy(np.array(list(context[1]))).float().to(device)
        
        def _factT(m):
            return np.power(m+1, 1/3)
        mu1 = self.func_1(tensor1)
        mu2 = self.func_2(tensor2)
        tensor3 = torch.squeeze(torch.stack([mu1,mu2], dim=2))
        
        mu3 = self.func_3(tensor3)
                #print("mu:", len(mu), len(mu[0]))
        g_list_1 = []
        g_list_2 = []
        g_list_3 = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        for fx_1, fx_2, fx_3 in zip(mu1, mu2, mu3):
            self.func_1.zero_grad()
            self.func_2.zero_grad()
            self.func_3.zero_grad()
            
            fx_1.backward(retain_graph=True)
            fx_2.backward(retain_graph=True)
            fx_3.backward(retain_graph=True)
            
            g_1 = torch.cat([p.grad.flatten().detach() for p in self.func_1.parameters()])
            g_2 = torch.cat([p.grad.flatten().detach() for p in self.func_2.parameters()])
            g_3 = torch.cat([p.grad.flatten().detach() for p in self.func_3.parameters()])
            
            
            g_list_1.append(g_1)
            g_list_2.append(g_2)
            g_list_3.append(g_3)
            
            sigma_1 = torch.sqrt(torch.sum(self.lamdba * g_1 * g_1 / self.U_1)).item()
            sigma_2 = torch.sqrt(torch.sum(self.lamdba * g_2 * g_2 / self.U_2)).item()
            sigma_3 = torch.sqrt(torch.sum(self.lamdba * g_3 * g_3 / self.U_3)).item()
            sum_sigma = (sigma_1+ sigma_2 +sigma_3)/3

            sample_r = fx_3.item() +  sum_sigma
            sampled.append(sample_r)
            ave_sigma += sum_sigma
            ave_rew += sample_r
        arm = np.argmax(sampled)
        #print("arm:", arm)
        self.U_1 += g_list_1[arm] * g_list_1[arm]
        self.U_2 += g_list_2[arm] * g_list_2[arm]
        self.U_3 += g_list_3[arm] * g_list_3[arm]
        return arm, g_list_3[arm].norm().item(), ave_sigma, ave_rew

    def train(self, context,  final_r, subrewards, t):
       
        self.context_list.append(context)
        self.reward.append((subrewards[0], subrewards[1], final_r)) 
                
        optimizer_1 = optim.SGD(self.func_1.parameters(), lr=1e-2, weight_decay=self.lamdba)
        optimizer_2 = optim.SGD(self.func_2.parameters(), lr=1e-2, weight_decay=self.lamdba)
        optimizer_3 = optim.SGD(self.func_3.parameters(), lr=1e-2, weight_decay=self.lamdba)


        length = len(self.context_list)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c1  = self.context_list[idx][0]
                c2  = self.context_list[idx][1]
                c1 = torch.from_numpy(c1).float().to(device)
                c2 = torch.from_numpy(c2).float().to(device)
                r1 = self.reward[idx][0]
                r2 = self.reward[idx][1]
               
                optimizer_1.zero_grad()
                optimizer_2.zero_grad()
                optimizer_3.zero_grad()

                mid_1 = self.func_1(c1)
                delta_1 = mid_1 - r1
                loss_1 = delta_1 ** 2
 
                
                mid_2 = self.func_2(c2)
                delta_2 = mid_2 - r2
                loss_2 = delta_2 ** 2
        
                
                c3 =torch.squeeze(torch.dstack((mid_1,mid_2)))
                r3 = self.reward[idx][2]
               
                mid_3 = self.func_3(c3)
                delta_3 = mid_3 - r3
                loss_3 = delta_3 ** 2
                loss_3.backward()
                optimizer_1.step()
                optimizer_2.step()
                optimizer_3.step()
      
                
                loss = loss_1.item() + loss_2.item() + loss_3.item()
                batch_loss += loss
                tot_loss += loss
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length
                
                
        
                


