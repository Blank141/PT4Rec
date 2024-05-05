import json
from turtle import forward
import torch
torch.manual_seed(12345)
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.augmentor import GraphAugmentor
from torchnmf.nmf import NMF
import numpy as np
from model.graph.XSimGCL import XSimGCL_Encoder
from model.graph.SimGCL import SimGCL_Encoder


# 重写prompts生成方式 train和save
class CPMP(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(CPMP, self).__init__(conf, training_set, test_set)

        args = OptionConf(self.config['CPMP'])
        self.n_layers = int(args['-n_layer'])
        temp = float(args['-temp'])
        prompt_size = int(args['-prompt_size'])
        self.user_prompt_num = int(args['-user_prompt_num'])
        self.pretrain_model = args['-pretrain_model']

        self.user_prompt_H = True
        self.user_prompt_M = True
        self.user_prompt_R = True

        if self.pretrain_model == 'XSimGCL':
            self.eps = 0.2
            self.layer_cl = 1
            self.model = XSimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.layer_cl, temp)
        elif self.pretrain_model == 'SimGCL':
            self.model = SimGCL_Encoder(self.data, self.emb_size, eps=0.1, n_layers=3)

        if self.user_prompt_num != 0:         
            self.user_prompt_generator = [Prompts_Generator(self.emb_size, prompt_size).cuda() for _ in range(self.user_prompt_num)]
            self.user_attention = Attention(prompt_size, self.user_prompt_num).cuda()

        self.interaction_mat = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat).cuda()
        self.user_matrix, self.Item_matrix = self._adjacency_matrix_factorization()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj).cuda()

    def XSimGCL_pre_train(self):
        # 若已有预训练模型，则直接加载
        # try:
        #     self.model.load_state_dict(torch.load('./pretrained_model/XSimGCL_Douban_pretrain_20.pt'))
        #     print('############## Pre-Training Phase ##############')
        #     print('Load pretrained model successfully!')
        #     return
        # except:
        #     print('No pretrained model, start pre-training...')

        pre_trained_model = self.model.cuda()
        optimizer = torch.optim.Adam(pre_trained_model.parameters(), lr=self.lRate)

        print('############## Pre-Training Phase ##############')
        for epoch in range(self.maxPreEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb  = pre_trained_model(True)
                cl_loss = pre_trained_model.cal_cl_loss([user_idx,pos_idx],rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb)
                batch_loss =  cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0:
                    print('pre-training:', epoch + 1, 'batch', n, 'cl_loss', cl_loss.item())

        # save pre-trained model
        # torch.save(pre_trained_model.state_dict(), './pretrained_model/XSimGCL_Douban_pretrain_20.pt')         

    def SimGCL_pre_train(self):
        # 若已有预训练模型，则直接加载
        try:
            self.model.load_state_dict(torch.load('./pretrained_model/SimGCL_douban_pretrain_20.pt'))
            print('############## Pre-Training Phase ##############')
            print('Load pretrained model successfully!')
            return
        except:
            print('No pretrained model, start pre-training...')

        pre_trained_model = self.model.cuda()
        optimizer = torch.optim.Adam(pre_trained_model.parameters(), lr=self.lRate)

        print('############## Pre-Training Phase ##############')
        for epoch in range(self.maxPreEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                cl_loss = pre_trained_model.cal_cl_loss([user_idx,pos_idx])
                batch_loss =  cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0:
                    print('pre-training:', epoch + 1, 'batch', n, 'cl_loss', cl_loss.item())

        # save pre-trained model
        torch.save(pre_trained_model.state_dict(), './pretrained_model/SimGCL_gowalla_pretrain_20.pt')    

    def _csr_to_pytorch_dense(self, csr):
        array = csr.toarray()
        dense = torch.Tensor(array)
        return dense.cuda()

    def user_historical_records(self, item_emb):    # 用户的历史记录 根据所有选择过该用户的商品的embedding
        user_profiles = torch.mm(self.interaction_mat, item_emb)
        return user_profiles

    def _adjacency_matrix_factorization(self): # 邻接矩阵分解
        adjacency_matrix = self.data.interaction_mat
        adjacency_matrix = adjacency_matrix.toarray()
        adjacency_matrix = torch.Tensor(adjacency_matrix).cuda().t()

        print('######### Adjacency Matrix Factorization #############')
        nmf = NMF(Vshape=adjacency_matrix.shape, rank=self.emb_size).cuda() # torch
        # user_profiles = torch.Tensor(nmf.W).cuda()
        # item_profiles = torch.Tensor(nmf.H).cuda()
        user_profiles = nmf.W
        item_profiles = nmf.H
        return user_profiles, item_profiles

    def _high_order_relations(self, item_emb, user_emb):  # 高阶关系
        # small dataset
        # emb = torch.cat((user_emb, item_emb), 0)
        # inputs = torch.sparse.mm(self.ui_high_order, emb)
        # inputs = inputs[:self.data.user_num, :]
        # return inputs

        # big dataset Ciao
        ego_embeddings = torch.cat((user_emb, item_emb), 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_profiles, item_profiles = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_profiles, item_profiles

    def train(self):
        if self.pretrain_model == 'XSimGCL':
            self.XSimGCL_pre_train()
        elif self.pretrain_model == 'SimGCL':
            self.SimGCL_pre_train()

        model = self.model.cuda()
        params = list(model.parameters()) + list(self.user_attention.parameters()) + list(self.user_prompt_generator[0].parameters()) + list(self.user_prompt_generator[1].parameters()) + list(self.user_prompt_generator[2].parameters())
        optimizer = torch.optim.Adam(params, lr=self.lRate)

        metrics =[]
        print('############## Downstream Training Phase ##############')
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_emb, item_emb = model()

                prompted_user_emb, prompted_item_emb = self.generate_prompts(user_emb, item_emb)

                user_idx, pos_idx, neg_idx = batch
                # rec_user_emb, rec_item_emb = model()
                rec_user_emb = prompted_user_emb
                rec_item_emb = prompted_item_emb

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)

                # 第二阶段 不继续训练对比学习权重
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb)
                # Backward and optimize
                
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if n % 100==0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item())#, 'cl_loss', cl_loss.item())
            with torch.no_grad():
                user_emb, self.item_emb = self.model()
                prompted_user_emb, prompted_item_emb = self.generate_prompts(user_emb, self.item_emb)
                self.user_emb = prompted_user_emb
                self.item_emb = prompted_item_emb
            if epoch%5==0:
                metric = self.fast_evaluation(epoch)
                metrics.append(metric)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        # metrics 存为json文件
        # with open('CPMP'+ str(self.maxPreEpoch) +'_metrics.json', 'w') as f:
        #     json.dump(metrics, f)

        #### 计算注意力权重并保存
        # alpha = torch.matmul(self.user_emb, self.user_attention.A)
        # alpha = F.softmax(alpha/8, dim=1)
        # alpha = alpha.cpu().detach().numpy()
        # np.save('XSimGCL_Douban_attention_weight.npy', alpha)

    def save(self):
        with torch.no_grad():
            user_emb, item_emb = self.model.forward()
            prompted_user_emb, prompted_item_emb = self.generate_prompts(user_emb, item_emb)
            self.best_user_emb = prompted_user_emb
            self.best_item_emb = prompted_item_emb

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()
    
    def generate_prompts(self, user_emb, item_emb):
        user_prompts = []
        u = 0
        if self.user_prompt_num != 0:
            if self.user_prompt_H:
                user_prompts.append(self.user_prompt_generator[u](self.user_historical_records(item_emb)))
                u += 1
            if self.user_prompt_M:
                user_prompts.append(self.user_prompt_generator[u](self.user_matrix))
                u += 1
            if self.user_prompt_R:
                user_prompts.append(self.user_prompt_generator[u](self._high_order_relations(item_emb, user_emb)[0]))
            # 注意力
            user_prompt = torch.stack(user_prompts, dim=1)
            prompt = self.user_attention(user_prompt, user_emb)
            prompted_user_emb = prompt * user_emb
        else:
            prompted_user_emb = user_emb

        return prompted_user_emb, item_emb


class Attention(nn.Module):
    def __init__(self, prompt_size, prompt_num):
        super(Attention, self).__init__()
        initializer = nn.init.xavier_uniform_
        self.prompt_num = prompt_num
        self.A = nn.Parameter(initializer(torch.empty(prompt_size, prompt_num)))

    def forward(self, prompt_base, embeddings):
        alpha = torch.matmul(embeddings, self.A)
        alpha = F.softmax(alpha, dim=1)
        alpha = alpha.unsqueeze(1)  # alpha.shape = [2829, 1, 3]
        prompt = torch.bmm(alpha, prompt_base)  # prompt.shape = [2829, 1, 64]
        prompt = prompt.squeeze(1)  # prompt.shape = [2829, 64]
        return prompt

    
class Prompts_Generator(nn.Module):
    def __init__(self, emb_size, prompt_size):
        super(Prompts_Generator, self).__init__()
        self.W = nn.Linear(emb_size, prompt_size)
        self.activation = nn.Tanh()

    def forward(self, inputs):
        prompts = inputs
        prompts = self.W(prompts)
        prompts = self.activation(prompts)
        
        return prompts
    