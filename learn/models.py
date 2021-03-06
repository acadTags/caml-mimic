"""
    Holds PyTorch models
"""
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
from torch.autograd import Variable

import numpy as np

from math import floor
import random
import sys
import time

from constants import *
from dataproc import extract_wvs

class BaseModel(nn.Module):

    def __init__(self, Y, embed_file, dicts, lmbda=0, dropout=0.5, gpu=True, embed_size=100, lmbda_sim=0, lmbda_sub=0):
        super(BaseModel, self).__init__()
        torch.manual_seed(1337)
        self.gpu = gpu
        self.Y = Y
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)
        self.lmbda = lmbda
        self.lmbda_sim = lmbda_sim
        self.lmbda_sub = lmbda_sub

        #make embedding layer
        if embed_file:
            print("loading pretrained embeddings...")
            W = torch.Tensor(extract_wvs.load_embeddings(embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            #add 2 to include UNK and PAD
            vocab_size = len(dicts['ind2w'])
            self.embed = nn.Embedding(vocab_size+2, embed_size, padding_idx=0) # random initialisation
            

    def _get_loss(self, yhat, target, diffs=None, sim_reg=None, sub_reg=None):
        #calculate the BCE
        loss = F.binary_cross_entropy_with_logits(yhat, target)
        # torch.nn.BCEWithLogitsLoss(weight=None, size_average=True)https://pytorch.org/docs/0.3.1/nn.html?highlight=binary_cross_entropy_with_logits#torch.nn.BCEWithLogitsLoss
        
        #add description regularization loss if relevant
        if self.lmbda > 0 and diffs is not None:
            diff = torch.stack(diffs).mean() 
            #about torch.stack() https://pytorch.org/docs/0.3.1/torch.html?highlight=torch%20stack#torch.stack -HD
            #about torch.mean() https://pytorch.org/docs/0.3.1/torch.html#torch.mean -HD
            loss = loss + diff
            
        #add sim loss and sub loss if relevant
        if self.lmbda_sim > 0 and sim_reg is not None:
            loss = loss + sim_reg
        if self.lmbda_sub > 0 and sub_reg is not None:
            loss = loss + sub_reg
        return loss

    def embed_descriptions(self, desc_data, gpu):
        #label description embedding via convolutional layer
        #number of labels is inconsistent across instances, so have to iterate over the batch
        b_batch = []
        for inst in desc_data:
            if len(inst) > 0:
                if gpu:
                    lt = Variable(torch.cuda.LongTensor(inst))
                else:
                    lt = Variable(torch.LongTensor(inst))
                d = self.desc_embedding(lt)
                d = d.transpose(1,2)
                d = self.label_conv(d)
                d = F.max_pool1d(F.tanh(d), kernel_size=d.size()[2])
                d = d.squeeze(2)
                b_inst = self.label_fc1(d)
                b_batch.append(b_inst)
            else:
                b_batch.append([])
        return b_batch

    def _compare_label_embeddings(self, target, b_batch, desc_data):
        #description regularization loss 
        #b is the embedding from description conv
        #iterate over batch because each instance has different # labels
        diffs = []
        for i,bi in enumerate(b_batch):
            ti = target[i]
            inds = torch.nonzero(ti.data).squeeze().cpu().numpy()

            zi = self.final.weight[inds,:]
            diff = (zi - bi).mul(zi - bi).mean()

            #multiply by number of labels to make sure overall mean is balanced with regard to number of labels
            diffs.append(self.lmbda*diff*bi.size()[0])
        return diffs
    
    #todo: add semantic-based loss regularization [soon]
    def _calcultate_semantic_based_lossreg(self,):
        return "" 
        
class BOWPool(BaseModel):
    """
        Logistic regression model over average or max-pooled word vector input
    """

    def __init__(self, Y, embed_file, lmbda, gpu, dicts, pool='max', embed_size=100, dropout=0.5, code_emb=None):
        super(BOWPool, self).__init__(Y, embed_file, dicts, lmbda, dropout=dropout, gpu=gpu, embed_size=embed_size)
        self.final = nn.Linear(embed_size, Y)
        #for nn.Linear see https://pytorch.org/docs/0.3.1/nn.html?highlight=nn%20linear#torch.nn.Linear
        #the embed_size and Y define the weight matrix size.
        if code_emb:
            self._code_emb_init(code_emb, dicts)
        else:
            xavier_uniform(self.final.weight)
        self.pool = pool
    
    #initialisation of the weight size as the code embeddings. -HD
    def _code_emb_init(self, code_emb, dicts):
        code_embs = KeyedVectors.load_word2vec_format(code_emb)
        #classmethod load_word2vec_format(fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict', limit=None, datatype=<class 'numpy.float32'>) 
        #Load the input-hidden weight matrix from the original C word2vec-tool format.
        weights = np.zeros(self.final.weight.size())
        for i in range(self.Y):
            code = dicts['ind2c'][i]
            weights[i] = code_embs[code]
        self.final.weight.data = torch.Tensor(weights).clone() # set weight as the code embeddings.

    def forward(self, x, target, desc_data=None, get_attention=False):
        #get embeddings and apply dropout
        x = self.embed(x)
        #x = self.embed_drop(x) #also applying dropout here for logistic regression. -HD
        #print('x', x) # to check the type of x. -HD
        #x = x.transpose(0, 2)
        #print('x-transposed', x) # to check x. -HD
        if self.pool == 'max':
            import pdb; pdb.set_trace() # this is for debugging -HD
            x = F.max_pool1d(x)
        else:
            #x = F.avg_pool1d(x) # TypeError: avg_pool1d() missing 1 required positional argument: 'kernel_size'
            #x = F.avg_pool1d(x, kernel_size=x.size()[2])
            x = torch.mean(x,1)
            #print('x-avg_pool1d',x)
        logits = F.sigmoid(self.final(x)) # only using the pooled, document embedding for logistic regression. In this case, it is also possible to apply SVM for the task. -HD
        #loss = self._get_loss(logits, target, diffs)
        loss = self._get_loss(logits, target, diffs=desc_data)
        return logits, loss, None

class ConvAttnPool(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, lmbda, gpu, dicts, embed_size=100, dropout=0.5, code_emb=None):
        super(ConvAttnPool, self).__init__(Y, embed_file, dicts, lmbda, dropout=dropout, gpu=gpu, embed_size=embed_size)

        #initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=int(floor(kernel_size/2))) # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) -HD
        #see https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/ about CNN and zeropadding
        xavier_uniform(self.conv.weight)

        #context vectors for computing attention as in 2.2
        self.U = nn.Linear(num_filter_maps, Y)
        #xavier_uniform(self.U.weight)

        #final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(num_filter_maps, Y)
        #xavier_uniform(self.final.weight)

        #initialize with trained code embeddings if applicable
        if code_emb:
            self._code_emb_init(code_emb, dicts)
            ##also set conv weights to do sum of inputs
            #weights = torch.eye(self.embed_size).unsqueeze(2).expand(-1,-1,kernel_size)/kernel_size
            #self.conv.weight.data = weights.clone()
            #self.conv.bias.data.zero_()
        else:
            xavier_uniform(self.U.weight)
            xavier_uniform(self.final.weight)
        #conv for label descriptions as in 2.5
        #description module has its own embedding and convolution layers
        if lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=int(floor(kernel_size/2)))
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform(self.label_fc1.weight)

    # def _code_emb_init(self, code_emb, dicts):
        # #code_embs = KeyedVectors.load_word2vec_format(code_emb)
        # code_embs = Word2Vec.load(code_emb)
        # weights = np.zeros(self.final.weight.size())
        # for i in range(self.Y):
            # code = dicts['ind2c'][i]
            # weights[i] = code_embs.wv[code]
        # self.U.weight.data = torch.Tensor(weights).clone()
        # self.final.weight.data = torch.Tensor(weights).clone()
    def _code_emb_init(self, code_emb, dicts):
        #code_embs = KeyedVectors.load_word2vec_format(code_emb)
        code_embs = Word2Vec.load(code_emb)
        print(self.Y, code_embs.vector_size)
        bound = np.sqrt(6.0) / np.sqrt(self.Y + code_embs.vector_size)  # bound for random variables for Xavier initialization.
        weights = np.zeros(self.final.weight.size())
        n_exist, n_inexist = 0, 0
        for i in range(self.Y):
            code = dicts['ind2c'][i]
            # the normalisation steps here follow brightmart's implementation, see def assign_pretrained_word_embedding(...) in 
            #https://github.com/brightmart/text_classification/blob/master/a02_TextCNN/p7_TextCNN_train.py
            if code in code_embs.wv.vocab:
                n_exist = n_exist + 1
                #weights[i] = code_embs.wv[code]
                vec = code_embs.wv[code]
                weights[i] = vec / float(np.linalg.norm(vec) + 1e-6) #normalise to unit length -HD #any better normalisation methods?
            else:
                n_inexist = n_inexist + 1
                weights[i] = np.random.uniform(-bound, bound, code_embs.vector_size); #using the original xavier uniform initialization. -HD
        print("code exists embedding:", n_exist, " ;code not exist embedding:", n_inexist)
        self.U.weight.data = torch.Tensor(weights).clone() # we want that similar labels attend to similar sets of ngrams in a document.
        self.final.weight.data = torch.Tensor(weights).clone() # we want that similar labels have similar output values in the prediction.
        print("final layer and attention layer: code embedding initialized")
        
    def forward(self, x, target, desc_data=None, get_attention=True, sim_data=None, sub_data=None):
        #get embeddings and apply dropout
        x = self.embed(x)
        x = self.embed_drop(x)
        #print('x',x.shape)
        x = x.transpose(1, 2)
        #print('x-transposed',x.shape)
        
        #apply convolution and nonlinearity (tanh)
        x = F.tanh(self.conv(x).transpose(1,2))
        #print('x-conv-transposed-nonlinearity',x.shape)
        
        #apply attention
        #print('self.U.weight',self.U.weight.shape)
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1,2)), dim=2)
        #print('alpha',alpha.shape) #[torch.cuda.FloatTensor of size 16x8921x118 (GPU 0)] #this is really a large size of alpha! -HD
        #document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)
        #print('m',m.shape) #[torch.cuda.FloatTensor of size 16x8921x50 (GPU 0)]
        
        #print('self.final.weight',self.final.weight.shape)
        #final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        #print('y',y) #[torch.cuda.FloatTensor of size 16x8921 (GPU 0)]

        #an example here
        #x torch.Size([16, 117, 100])
        #x-transposed torch.Size([16, 100, 117])
        #x-conv-transposed-nonlinearity torch.Size([16, 118, 50])
        #self.U.weight torch.Size([8921, 50])
        #alpha torch.Size([16, 8921, 118])
        #m torch.Size([16, 8921, 50])
        #self.final.weight torch.Size([8921, 50])
        #y Variable containing:
        # 8.2350e-02  1.1934e-01  1.3893e-01  ...   2.6954e-02  5.3924e-03  1.8069e-02
        # 8.2866e-02  1.2009e-01  1.3988e-01  ...   2.6592e-02  5.1176e-03  1.8395e-02
        # 8.1059e-02  1.1930e-01  1.3908e-01  ...   2.6202e-02  5.7202e-03  1.7288e-02
                       # ...                   ⋱                   ...
        # 8.2959e-02  1.1797e-01  1.3966e-01  ...   2.5005e-02  7.7955e-03  1.8791e-02
        # 8.3862e-02  1.1899e-01  1.3727e-01  ...   2.7737e-02  9.2247e-03  1.8746e-02
        # 8.4048e-02  1.1827e-01  1.3769e-01  ...   2.5353e-02  8.2030e-03  1.8507e-02
        #[torch.cuda.FloatTensor of size 16x8921 (GPU 0)]

        if desc_data is not None:
            #run descriptions through description module
            b_batch = self.embed_descriptions(desc_data, self.gpu)
            #get l2 similarity loss
            diffs = self._compare_label_embeddings(target, b_batch, desc_data)
        else:
            diffs = None
            
        #final sigmoid to get predictions
        yhat = y
        loss = self._get_loss(yhat, target, diffs)
        return yhat, loss, alpha


class VanillaConv(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, gpu=True, dicts=None, embed_size=100, dropout=0.5, code_emb=None):
        super(VanillaConv, self).__init__(Y, embed_file, dicts, dropout=dropout, embed_size=embed_size) 
        #initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size)
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) -HD
        xavier_uniform(self.conv.weight)

        #linear output
        self.fc = nn.Linear(num_filter_maps, Y)
        
        print('code_emb',code_emb)
        #initialize with trained code embeddings if applicable
        if code_emb: # if a code embedding exists for a label, then initialize the self.fc.weight with code embedding.
            self._code_emb_init(code_emb, dicts)
            ##also set conv weights to do sum of inputs
            #weights = torch.eye(self.embed_size).unsqueeze(2).expand(-1,-1,kernel_size)/kernel_size
            #self.conv.weight.data = weights.clone()
            #self.conv.bias.data.zero_()
        else:
            # otherwise, initialize the weight using xavier uniform.      
            xavier_uniform(self.fc.weight)  
            #print("final layer: xavier uniform initialized")
    
    def _code_emb_init(self, code_emb, dicts):
        #code_embs = KeyedVectors.load_word2vec_format(code_emb)
        code_embs = Word2Vec.load(code_emb)
        print(self.Y, code_embs.vector_size)
        bound = np.sqrt(6.0) / np.sqrt(self.Y + code_embs.vector_size)  # bound for random variables for Xavier initialization.
        weights = np.zeros(self.fc.weight.size())
        n_exist, n_inexist = 0, 0
        for i in range(self.Y):
            code = dicts['ind2c'][i]
            if code in code_embs.wv.vocab:
                n_exist = n_exist + 1
                #weights[i] = code_embs.wv[code]
                vec = code_embs.wv[code]
                weights[i] = vec / float(np.linalg.norm(vec) + 1e-6) #normalise to unit length -HD
            else:
                n_inexist = n_inexist + 1
                weights[i] = np.random.uniform(-bound, bound, code_embs.vector_size);
                #using the original xavier uniform initialization.
        print("code exists embedding:", n_exist, " ;code not exist embedding:", n_inexist)
        self.fc.weight.data = torch.Tensor(weights).clone()
        print("final layer: code embedding initialized")
        
    def forward(self, x, target, desc_data=None, get_attention=False):
        #print('calling the forward function now')
        #embed
        x = self.embed(x)
        x = self.embed_drop(x)
        #print('x',x.shape) # (batch_size,doc_length,embed_size)
        x = x.transpose(1, 2)
        #print('x',x.shape) # (batch_size,embed_size,doc_length)
        #conv/max-pooling
        c = self.conv(x)
        #print('c',c.shape) # (batch_size,num_filter_maps,(doc_length-kernel_size+1)/stride)
        if get_attention:
            #get argmax vector too
            x, argmax = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2], return_indices=True)
            attn = self.construct_attention(argmax, c.size()[2]) # 'fake' attention from the vanilla CNN for explanation -HD
        else:
            x = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2])
            #print('x-pooled',x.shape)
            attn = None
        x = x.squeeze(dim=2)
        #print('x-squeezed',x.shape)

        #linear output
        x = self.fc(x)
        #print('x-final',x.shape)
        
        #one example here
        #x torch.Size([16, 100, 392])
        #c torch.Size([16, 500, 389])
        #after pooling torch.Size([16, 500, 1])
        #after squeezing torch.Size([16, 500])
        #final torch.Size([16, 8921])

        #final sigmoid to get predictions
        yhat = x
        loss = self._get_loss(yhat, target)
        return yhat, loss, attn

    def construct_attention(self, argmax, num_windows):
        attn_batches = []
        for argmax_i in argmax:
            attns = []
            for i in range(num_windows):
                #generate mask to select indices of conv features where max was i
                mask = (argmax_i == i).repeat(1,self.Y).t()
                #apply mask to every label's weight vector and take the sum to get the 'attention' score
                weights = self.fc.weight[mask].view(-1,self.Y)
                if len(weights.size()) > 0:
                    window_attns = weights.sum(dim=0)
                    attns.append(window_attns)
                else:
                    #this window was never a max
                    attns.append(Variable(torch.zeros(self.Y)).cuda())
            #combine
            attn = torch.stack(attns)
            attn_batches.append(attn)
        attn_full = torch.stack(attn_batches)
        #put it in the right form for passing to interpret
        attn_full = attn_full.transpose(1,2)
        return attn_full


class VanillaRNN(BaseModel):
    """
        General RNN - can be LSTM or GRU, uni/bi-directional
    """

    def __init__(self, Y, embed_file, dicts, rnn_dim, cell_type, num_layers, gpu, embed_size=100, bidirectional=False):
        super(VanillaRNN, self).__init__(Y, embed_file, dicts, embed_size=embed_size, gpu=gpu)
        self.gpu = gpu
        self.rnn_dim = rnn_dim
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        #recurrent unit
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(self.embed_size, floor(self.rnn_dim/self.num_directions), self.num_layers, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(self.embed_size, floor(self.rnn_dim/self.num_directions), self.num_layers, bidirectional=bidirectional)
        #linear output
        self.final = nn.Linear(self.rnn_dim, Y)

        #arbitrary initialization
        self.batch_size = 16
        self.hidden = self.init_hidden()

    def forward(self, x, target, desc_data=None, get_attention=False):
        #clear hidden state, reset batch size at the start of each batch
        self.refresh(x.size()[0])

        #embed
        embeds = self.embed(x).transpose(0,1)
        #apply RNN
        out, self.hidden = self.rnn(embeds, self.hidden)

        #get final hidden state in the appropriate way
        last_hidden = self.hidden[0] if self.cell_type == 'lstm' else self.hidden
        last_hidden = last_hidden[-1] if self.num_directions == 1 else last_hidden[-2:].transpose(0,1).contiguous().view(self.batch_size, -1)
        #apply linear layer and sigmoid to get predictions
        yhat = self.final(last_hidden)
        loss = self._get_loss(yhat, target)
        return yhat, loss, None

    def init_hidden(self):
        if self.gpu:
            h_0 = Variable(torch.cuda.FloatTensor(self.num_directions*self.num_layers, self.batch_size,
                                                  floor(self.rnn_dim/self.num_directions)).zero_())
            if self.cell_type == 'lstm':
                c_0 = Variable(torch.cuda.FloatTensor(self.num_directions*self.num_layers, self.batch_size,
                                                      floor(self.rnn_dim/self.num_directions)).zero_())
                return (h_0, c_0)
            else:
                return h_0
        else:
            h_0 = Variable(torch.zeros(self.num_directions*self.num_layers, self.batch_size, floor(self.rnn_dim/self.num_directions)))
            if self.cell_type == 'lstm':
                c_0 = Variable(torch.zeros(self.num_directions*self.num_layers, self.batch_size, floor(self.rnn_dim/self.num_directions)))
                return (h_0, c_0)
            else:
                return h_0

    def refresh(self, batch_size):
        self.batch_size = batch_size
        self.hidden = self.init_hidden()
