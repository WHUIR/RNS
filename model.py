import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import activation_getter
class RNS(nn.Module):
    def load_word2vec(self, vocabulary):
        word_vec_path = 'data/glove_twitter_27B/glove25d.txt'
        with open(word_vec_path, "r", encoding='UTF-8') as f:
            lines = f.readlines()
        raw_word2vec = {}
        for l in lines:
            w, vec = l.split(" ", 1)
            raw_word2vec[w] = vec
        self.word2vec = None
        oov_cnt = 0
        for word, index in vocabulary.items():
            str_vec = raw_word2vec.get(word, None)
            if str_vec is None:
                oov_cnt += 1
                vec = np.random.randn(25) * 0.1
            else:
                vec = np.fromstring(str_vec, sep=" ")
            vec = np.expand_dims(vec, axis=0)
            self.word2vec = np.concatenate((self.word2vec, vec), 0) if self.word2vec is not None else vec
        print("word2vec cannot cover %f vocabulary" % (float(oov_cnt) / len(vocabulary)))
    def __init__(self, num_users, num_items, model_args, u_text, i_text, vocabulary):
        super(RNS, self).__init__()
        self.args = model_args
        L = self.args.L
        self.n_t = self.args.nt
        self.n_k = self.args.nk
        self.drop_ratio = self.args.drop
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]
        self.num_users = num_users
        self.num_items = num_items
        self.word2vec = None
        self.w_vec_dim = self.args.dim
        self.alpha = self.args.alpha
        self.word_embed = nn.Embedding(len(vocabulary), self.w_vec_dim)
        self.word_embed.weight.data.normal_(0, 1.0 / self.word_embed.embedding_dim)
        self.word_embed.weight.requires_grad = True
        self.u_text_fake_embed = nn.Embedding(u_text.shape[0], u_text.shape[1])
        self.i_text_fake_embed = nn.Embedding(i_text.shape[0], i_text.shape[1])
        self.u_text_fake_embed.weight.data.copy_(torch.from_numpy(u_text))
        self.u_text_fake_embed.weight.requires_grad = False
        self.i_text_fake_embed.weight.data.copy_(torch.from_numpy(i_text))
        self.i_text_fake_embed.weight.requires_grad = False
        self.trans_mat = [torch.randn((self.w_vec_dim, self.w_vec_dim), dtype=torch.float, requires_grad=True).to('cuda')
                          for _ in range(self.n_k)]
        lengths = [1, 3, 5, 7, 9]
        self.conv_t_item = nn.ModuleList([nn.Conv2d(self.n_k, self.n_t, (i, self.w_vec_dim)) for i in lengths])
        self.conv_t_user = nn.ModuleList([nn.Conv2d(self.n_k, self.n_t, (i, self.w_vec_dim)) for i in lengths])
        self.k_dim = self.n_t * len(lengths)
        self.full_dim = self.n_k * self.k_dim
        self.pos_embed = nn.Embedding(L, self.k_dim)
        self.conv_d = nn.ModuleList([nn.Conv2d(1, 1, (L, 1)) for _ in range(self.n_k)])
        self.dropout = nn.Dropout(self.drop_ratio)
    def forward(self, seq_var, user_var, item_var, for_pred=False):
        seq_var_word_index = self.i_text_fake_embed(seq_var).long()
        seq_var_word_vector = self.word_embed(seq_var_word_index)
        item_var_word_index = self.i_text_fake_embed(item_var).long()
        item_var_word_vector = self.word_embed(item_var_word_index)
        user_var_word_index = self.u_text_fake_embed(user_var).long()
        user_var_word_vector = self.word_embed(user_var_word_index)
        l1, l2, l3, ll1, ll2, ll3 = [list() for _ in range(6)]
        seq_var_aspect_concat = None
        item_var_aspect_concat = None
        user_var_aspect_concat = None
        for i in range(self.n_k):
            seq_var_word_trans_vector = torch.einsum("abcd,de->abce", (seq_var_word_vector, self.trans_mat[i]))\
                .unsqueeze(1)
            seq_var_aspect_concat = torch.cat((seq_var_aspect_concat, seq_var_word_trans_vector), 1) \
                if seq_var_aspect_concat is not None else seq_var_word_trans_vector
            item_var_word_trans_vector = torch.einsum("abcd,de->abce", (item_var_word_vector, self.trans_mat[i])) \
                .unsqueeze(1)
            item_var_aspect_concat = torch.cat((item_var_aspect_concat, item_var_word_trans_vector), 1) \
                if item_var_aspect_concat is not None else item_var_word_trans_vector
            user_var_word_trans_vector = torch.einsum("abcd,de->abce", (user_var_word_vector, self.trans_mat[i])) \
                .unsqueeze(1)
            user_var_aspect_concat = torch.cat((user_var_aspect_concat, user_var_word_trans_vector), 1) \
                if user_var_aspect_concat is not None else user_var_word_trans_vector
        for j in range(seq_var_aspect_concat.shape[2]):
            s = seq_var_aspect_concat[:, :, j, :, :]
            for conv in self.conv_t_item:
                conv_out = self.ac_conv(conv(s).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                l1.append(pool_out)
            s1 = torch.cat(l1, 1).unsqueeze(1)  
            l2.append(s1)
            l1 = []
        seq_var_repr = torch.cat(l2, 1)
        for k in range(item_var.shape[1]):
            ss = item_var_aspect_concat[:, :, k, :, :]
            for conv in self.conv_t_item:
                conv_out = self.ac_conv(conv(ss).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                ll1.append(pool_out)
            ss1 = torch.cat(ll1, 1).unsqueeze(1)
            ll2.append(ss1)
            ll1 = []
        item_repr = torch.cat(ll2, 1)  
        seq_var_repr += self.pos_embed(torch.arange(0, seq_var.shape[1]).to("cuda"))
        l4 = []
        for conv in self.conv_t_user:
            conv_out = self.ac_conv(conv(user_var_aspect_concat.squeeze(2)).squeeze(3))
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            l4.append(pool_out)
        user_repr = torch.cat(l4, 1)    
        al = []
        for i in range(item_var.shape[1]):
            s = item_repr[:, i, :].unsqueeze(1)
            w = F.softmax(torch.sum(seq_var_repr * s, 2), 1).unsqueeze(2)   
            u = torch.sum(seq_var_repr * w, 1).unsqueeze(1)    
            m = torch.argmax(w, 1).unsqueeze(1)  
            index = m.expand(-1, 1, seq_var_repr.size(2))
            if for_pred:
                seq_var_repr = seq_var_repr.expand(item_var.size(0), -1, -1)
            p = seq_var_repr.gather(1, index)   
            p_u = torch.cat((u, p), 1)
            w2 = F.softmax(torch.sum(p_u * s, 2), 1).unsqueeze(2)
            ss = torch.sum(p_u * w2, 1).unsqueeze(1)
            al.append(ss)
        seq_repr = torch.cat(al, 1) 
        if for_pred:
            res = torch.sum((self.alpha * seq_repr + user_repr.unsqueeze(1)).squeeze() * item_repr.squeeze(), 1)
        else:
            res = torch.sum((self.alpha * seq_repr + user_repr.unsqueeze(1)) * item_repr, 2)
        return res