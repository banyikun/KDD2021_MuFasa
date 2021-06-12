import numpy as np 
import pickle
import os
import time
import pandas as pd 
import scipy as sp
from data_proc import Bandit_multi
from model import MuFasa
import itertools

def get_new_context(contexts):
    new_contexts = []
    dim = 0
    index_cont = []
    for cont in contexts:
        dim +=len(cont[0])
        index_cont.append(range(len(cont)))
    for ele in itertools.product(*contexts):
        new_contexts.append(ele)
    index = []
    for i in itertools.product(*index_cont):
        index.append(i)
    return np.array(new_contexts), index


def get_final_reward_2(subrewards):
    return subrewards[0]+subrewards[1]


def get_final_reward(subrewards):
    return sum(subrewards)


if __name__ == '__main__':
    arg_datasets = ['notmnist', 'mnist']
    arg_size = 1
    arg_shuffle = 1
    arg_seed = 0
    arg_nu = 1
    arg_lambda = 0.0001
    arg_hidden = 100
    arg_num_tasks = 2

    datasets = []
    for arg_dataset in arg_datasets:
        use_seed = None if arg_seed == 0 else arg_seed
        b = Bandit_multi(arg_dataset, is_shuffle=arg_shuffle, seed=use_seed)
        datasets.append(b)

    input_dim = []
    for b in datasets:
        input_dim.append(b.dim)

    bandit_info = '{}'.format(arg_dataset)
    l = MuFasa(input_dim, arg_num_tasks, arg_lambda, arg_nu, arg_hidden)

    regrets = []
    summ = [0 for i in range(arg_num_tasks)]

    for t in range(101):
        conts = []
        rwds = []
        for data in datasets:
            context, rwd = data.step()
            conts.append(context)
            rwds.append(rwd)

        org_context, index_cont = get_new_context(conts)
        new_context = np.transpose(org_context, (1, 0, 2))
        arm_select, nrm, sig, ave_rwd = l.select(new_context, t)
        arms = index_cont[arm_select]
        task = 0
        subrewards = []
        for arm in arms:
            r = rwds[task][arm]
            subrewards.append(r)
            reg = np.max(rwds[task]) - r
            summ[task]+=reg
            task+=1
        final_r = get_final_reward(subrewards)
        if t%2 == 0:
            loss = l.train(org_context[arm_select], final_r, subrewards, t)

        regrets.append([summ[i] for i in range(arg_num_tasks)])
        if t % 100 == 0:
            print('{}: {}, {:.3e}, {:.3e}, {:.3e}, {:.3e}'.format(t, summ, loss, nrm, sig, ave_rwd))

