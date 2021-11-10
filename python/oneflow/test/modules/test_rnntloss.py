"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
from collections import OrderedDict

import numpy as np
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest

def  forward_logsoftmax(acts):
    max_value = np.max(acts, -1, keepdims=True)
    exp = np.exp(acts - max_value)
    exp_sum = np.sum(exp, -1, keepdims=True)
    out = acts - max_value - np.log(exp_sum)
    return out,exp/exp_sum

def grad_logsoftmax(grad_out,grad):
    sum = np.sum(grad_out, -1, keepdims=True)
    tmp = np.multiply(sum,grad)
    return grad_out-tmp

def log_sum_exp(a,b):
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a>b:
        return np.log(np.exp(b-a)+1)+a
    else:
        return np.log(np.exp(a-b)+1)+b


def rnnt_forward(acts, labels, act_length, label_length, blank):
    batch,maxT,maxU,alpha_size = acts.shape

    grads = np.zeros(acts.shape).reshape(-1)
    costs = np.zeros(acts.shape[0])
    
    acts = acts.reshape(-1)
    labels = labels.reshape(-1)


    def cost_and_grad_kernel(act_loc,label_loc,mb,T,U):
       
        def id2(t,u):
            return t*U+u

        def id3(t,u,v):
            return (t * maxU + u) * alpha_size + v

        alphas = np.zeros(T*U)
        betas = np.zeros(T*U)
        log_prob2 = np.zeros(T*U*2)

        for t in range(T):
            for u in range(U):
                offset = (t*U+u) * 2
                log_prob2[offset] = acts[act_loc + id3(t,u,blank)]
                if u < U-1:
                    log_prob2[offset+1]=acts[act_loc + id3(t,u,labels[label_loc + u])]

        alphas[0] = 0
        for t in range(T):
            for u in range(U):
                if u==0 and t > 0:
                    alphas[id2(t,0)] = alphas[id2(t-1,0)] + log_prob2[id2(t-1,0)*2]
                if t==0 and u > 0:
                    alphas[id2(0,u)] = alphas[id2(0,u-1)] + log_prob2[id2(0, u-1) * 2 + 1]
                if t > 0 and u > 0:
                    no_emit = alphas[id2(t-1, u)] + log_prob2[id2(t-1, u) * 2]
                    emit = alphas[id2(t, u-1)] + log_prob2[id2(t, u-1) * 2 + 1]
                    alphas[id2(t, u)] = log_sum_exp(no_emit,emit)
        

        llForward = alphas[id2(T-1, U-1)] + log_prob2[id2(T-1, U-1) * 2]
        
        betas[id2(T-1, U-1)] = log_prob2[id2(T-1, U-1) * 2]
        
        for t in range(T-1,-1,-1):
            for u in range(U-1,-1,-1):
                if u==U-1 and t<T-1:
                    betas[id2(t, U-1)] = betas[id2(t+1, U-1)] + log_prob2[id2(t, U-1) * 2]
                if t==T-1 and u<U-1:
                    betas[id2(T-1, u)] = betas[id2(T-1, u+1)] + log_prob2[id2(T-1, u) * 2 + 1]
                if t<T-1 and u<U-1:
                    no_emit = betas[id2(t+1, u)] + log_prob2[id2(t, u) * 2]
                    emit = betas[id2(t, u+1)] + log_prob2[id2(t, u) * 2 + 1]
                    betas[id2(t, u)] = log_sum_exp(emit,no_emit)

        loglike = betas[0]
        for t in range(T):
            for u in range(U):
                if t<T-1:
                    g = alphas[id2(t, u)]+betas[id2(t+1, u)]
                    grads[act_loc + id3(t, u, blank)] = \
                        -np.exp(log_prob2[id2(t, u) * 2]+g-loglike)
                if u<U-1:
                    g = alphas[id2(t, u)] + betas[id2(t, u+1)]
                    grads[act_loc+id3(t, u, labels[label_loc+u])] = \
                        -np.exp(log_prob2[id2(t, u) * 2 + 1]+g-loglike)
        grads[act_loc+id3(T-1, U-1, blank)] = \
            -np.exp(log_prob2[id2(T-1, U-1) * 2] + alphas[id2(T-1, U-1)] - loglike)   

        return -llForward

    for i in range(batch):
        T = act_length[i]
        U = label_length[i] + 1
        batch_size = maxT * maxU * alpha_size
        costs[i] = cost_and_grad_kernel(i*batch_size, i*(maxU-1), i, T, U)

    return costs,grads.reshape(batch,maxT,maxU,alpha_size)

def rnnt_backward(grad_out,grad):
    grad_out = grad_out.reshape(-1,1,1,1)
    return np.multiply(grad_out,grad)


def compare_with_np(
    device_type,
    reduction,
):
    acts = np.random.rand(2,3,3,4).astype(np.float32)
    labels =  np.array([[1, 2],[2,2]],dtype=np.int32)
    act_length = np.array([2,2],dtype=np.int32)
    label_length = np.array([2,1],dtype=np.int32)
    
    acts_o = flow.tensor(acts,dtype=flow.float32,requires_grad=True,device=flow.device(device_type))
    labels_o =  flow.tensor(labels,dtype = flow.int32, device=flow.device(device_type))
    act_length_o = flow.tensor(act_length,dtype = flow.int32,device=flow.device(device_type))
    label_length_0 = flow.tensor(label_length,dtype=flow.int32,device=flow.device(device_type))
    
    rnnt = flow.nn.RNNTLoss(reduction).to(flow.device(device_type))
    loss = rnnt(acts_o,labels_o,act_length_o,label_length_0)
    loss.backward()

    act_log,grad_log = forward_logsoftmax(acts)
    costs_rnnt,grad_rnnt = rnnt_forward(act_log,labels,act_length,label_length,blank=0)

    rnnt_grad = rnnt_backward(np.ones_like(costs_rnnt),grad_rnnt)
    act_grad = grad_logsoftmax(rnnt_grad,grad_log)

    costs_rnnt = costs_rnnt.sum()
    if reduction == "mean":
        costs_rnnt = costs_rnnt/acts.shape[0]
        act_grad = act_grad/acts.shape[0]
    assert np.allclose(costs_rnnt, loss.numpy(), atol=1e-05)
    assert np.allclose(acts_o.grad.numpy(),act_grad,atol=1e-05)


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cuda", "cpu"]
    arg_dict["reduction"] = ["mean", "sum"]
    return GenArgList(arg_dict)

@flow.unittest.skip_unless_1n1d()
class TestRNNTLoss1n1d(flow.unittest.TestCase):
    def test_rnnt_loss(test_case):
        for arg in gen_arg_list():
            compare_with_np(*arg)

if __name__ == "__main__":
    unittest.main()


