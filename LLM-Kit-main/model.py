from torch import nn
import numpy as np
from transformers import AutoTokenizer,AutoModel,AutoConfig
from torch.nn import CrossEntropyLoss,L1Loss,Softmax
import torch
from torch.nn import init
from transformers import AdamW
from math import sqrt
import os
import json
from torch_geometric.data import Data as tudata
from torch_geometric.nn import GCNConv
#from torch_geometric.nn import GATConv as GCNConv
import torch.nn.functional as F

class graphModel(nn.Module):
    # currently, we only show Asp-KGs of restaurant. we will public all as soon as the paper is accepted.
    def __init__(self,bert,tokenizer,aspects_num,aspect2order,device,graphfile='KGRL/KGRL-main/graphs/',dim=768): 
        super(graphModel,self).__init__()
        self.graphfile = graphfile
        self.dim = dim
        self.bert = bert
        self.aspects_num = aspects_num
        self.tokenizer = tokenizer
        self.mid2name = {}
        self.graphinfo = {}
        #self.conv = GCNConv(dim,dim)
        #self.conv1 = GCNConv(dim,dim)
        self.act = nn.Tanh()
        self.device = device
        self.aspect2order = aspect2order
        
        mls = [GCNConv(dim,dim) for _ in range(2*aspects_num)]
        self.mls = nn.ModuleList(mls)

        for filename in os.listdir(graphfile):
            before, after = os.path.splitext(filename)
            if after == '.json':
                with open(self.graphfile+filename,'r') as f:
                    self.mid2name = json.load(f)
                for k in self.mid2name:
                    v = self.mid2name[k]
                    s = ''
                    for i in v:
                        s = s + i + ' '
                    s = s[:-1] 
                    self.mid2name[k] = s
        for filename in os.listdir(graphfile):
            before, after = os.path.splitext(filename)
            if after == '.txt':
                self.graphinfo[before] = self.buildgraphinfo(self.graphfile+filename)

    def buildgraphinfo(self,filename):
        '''
        return {
            graph:pyG data with no x,
            rootnodeid:[0,1,4,5,...],
            tokens:[
                [101,2,4,25,2],
                [101,2452,63],
                ...
            ],
        }
        '''
        edge_index = []
        rootnodeid = []
        tokens = []
        mid2id = {}
        id2name = {}

        idx = 0

        with open(filename,'r') as f:
            for i,line in enumerate(f):
                if i == 0:
                    line = line.strip().split()
                    for r in line:
                        if r not in mid2id.keys():
                            mid2id[r] = idx
                            idx += 1
                            rootnodeid.append(mid2id[r])
                        else:
                            rootnodeid.append(mid2id[r])
                else:
                    line = line.strip().split()[:-1]
                    for r in line:
                        if r not in mid2id.keys() and r[:2]=='m.':
                            mid2id[r] = idx
                            idx += 1
                    edge_index.append([mid2id[line[0]],mid2id[line[1]]])
                    edge_index.append([mid2id[line[1]],mid2id[line[0]]])

        for mid,id in mid2id.items():
            # print(mid,id)
            # print(len(self.mid2name))
            id2name[id] = self.mid2name[mid]

        for index in range(len(id2name)):
            tokens.append(id2name[index])
        
        tokens = self.tokenizer(tokens,padding='max_length',
                        max_length=3,truncation=True)['input_ids']
        
        return {
            'graph':tudata(edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous()),
            'rootnodeid':rootnodeid,
            'tokens':tokens
        }

    def __getitem__(self,key):
        '''
        input category name
        use bert model
        pass GNN
        return pooled root nodes
        '''
        if key in self.graphinfo.keys():
            graph = self.graphinfo[key]['graph']
            rootnodeid = self.graphinfo[key]['rootnodeid']
            tokens =torch.LongTensor( self.graphinfo[key]['tokens'])
            
            # tokens.cuda()

            # print('device',tokens.device)

            attrs = self.bert(tokens.to(self.device)).pooler_output
            graph.x = attrs
            
            # graph.cuda()
            x = graph.x
            x.cuda()
            e =  graph.edge_index
            e = e.to(x.device)
            # print(type(x.device))
            # print(x.device)
            # graph.x.cuda()
            # graph.edge_index.cuda()

            # print('---',x.device,e.device)
            #x = self.conv(graph.x,graph.edge_index)
            #x =self.conv1( self.act(self.conv(x,e)),e)
            #x = self.act(x)
            x = self.act(self.mls[self.aspect2order[key]-1](x,e))
           
            x = self.act(self.mls[self.aspect2order[key]-1+self.aspects_num](x,e))
            x = torch.mean(x[rootnodeid],dim=0)
            return x


class JointModel(nn.Module):
    def __init__(self, model_path, tokenizer_path, aspects_num, ignore_label,
               device, hidden_dropout_prob = 0.1, hidden_size = 768, 
                orderfile = 'KGRL/KGRL-main/processed/order.json', toload = False):
        super(JointModel,self).__init__()
        
        self.aspects_num = aspects_num
        self.hidden_size = hidden_size
        self.initbound = 1/sqrt(self.hidden_size)
        
        self.ce = CrossEntropyLoss(ignore_index=ignore_label)#ignore the label indicating non-reference
        self.l1 = L1Loss()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.softmax = Softmax(dim=-1)
        #self.w2 = None

        if not toload:
            self.bert = AutoModel.from_pretrained(model_path)
        else:
            modelconfig = AutoConfig.from_pretrained(model_path)
            self.bert = AutoModel.from_config(modelconfig)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        #self.gm = graphModel(self.bert,self.tokenizer,device)

        self.aspectorderlist = []
        with open(orderfile,'r') as f:
            aspect2order = json.load(f)
        order2aspect = {}
        for k,v in aspect2order.items():
            order2aspect[v] = k
        for i in range(len(aspect2order)):
            self.aspectorderlist.append(order2aspect[i+1])
        # this two is for rating predicton
        #self.beta = nn.Parameter(torch.randn(1,hidden_size),requires_grad=True)
        #self.linear4rp = nn.Linear(self.hidden_size,self.hidden_size,bias=True)
        self.gm = graphModel(self.bert,self.tokenizer,aspects_num,aspect2order,device)
        self.w1 = nn.Parameter(torch.empty(self.aspects_num,1,self.hidden_size,self.hidden_size),requires_grad=True)
        init.uniform_(self.w1,-self.initbound,self.initbound)
        
        #replace w2！！！
        
        self.w2 = nn.Parameter(torch.empty(self.aspects_num,1,1,self.hidden_size),requires_grad=True)
        init.uniform_(self.w2,-self.initbound,self.initbound)
        
        self.w3 = nn.Parameter(torch.empty(self.aspects_num,1,self.hidden_size,self.hidden_size),requires_grad=True)
        init.uniform_(self.w3,-self.initbound,self.initbound)
        self.w4 = nn.Parameter(torch.empty(self.aspects_num,1,3,self.hidden_size),requires_grad=True)
        init.uniform_(self.w4,-self.initbound,self.initbound)
        self.b4 = nn.Parameter(torch.empty(self.aspects_num,1,3,1),requires_grad=True)
        init.uniform_(self.b4,-self.initbound,self.initbound)
        self.sigmoid = nn.Sigmoid()
        self.transform = nn.Linear(self.hidden_size,self.hidden_size,bias=False)
        self.coef = nn.Parameter(1*torch.tensor(1.),requires_grad=True) 

    def get_optimizer(self,lr_small,lr_big):
        bert_params_ids = list(map(id,self.bert.parameters()))
        other_params = list(filter(lambda p:id(p) not in bert_params_ids,self.parameters()))

        optimizer_params = [
            {'params':self.bert.parameters(),'lr':lr_small,'eps':1e-8},
            {'params':other_params,'lr':lr_big,'eps':1e-8}
        ]
        optimizer = AdamW(optimizer_params)
        return optimizer
        
    def save_pretrained(self,save_path):
        torch.save(self.state_dict(),save_path)
    
    def load_pretrained(self,path):
        self.load_state_dict(torch.load(path))

    def tokenize(self,sents):
        input_ids = []
        attention_masks = []
        for sent in sents:
            encode_dict = self.tokenizer.encode_plus(sent,
                                        add_special_tokens=True,
                                        padding='max_length',
                                        max_length=135,
                                        return_attention_mask=True,
                                        return_tensors='pt',
                                        truncation=True)
            input_ids.append(encode_dict['input_ids'])
            attention_masks.append(encode_dict['attention_mask'])
        input_ids = torch.cat(input_ids,dim=0)
        attention_masks = torch.cat(attention_masks,dim=0)
        return input_ids,attention_masks

    def forward(self,input_ids,attention_masks,w2=None):
        output = self.bert(
            input_ids,
            attention_mask = attention_masks
        )
        #to use [CLS] token embedding to predict star rating
        #CLS_output = self.dropout(output.last_hidden_state[:,0,:])
        #CLS_output = self.dropout(output.pooler_output)
        #all hidden_state
        H = output.last_hidden_state

        #rating prediction
        #x = self.tanh(self.linear4rp(CLS_output))
        #rp = (self.beta@(x.T)).reshape(-1,)
        #graphpart
        if w2 is None:
            w2 = []
            for aspect in self.aspectorderlist:
                #print('-')
           
                w2.append(self.gm[aspect])
            w2 = torch.cat(w2,0).reshape(self.aspects_num,1,1,self.hidden_size)
        tw2 = self.transform(w2)

        #ACSA
        H = H.permute(0,2,1)  # do  H = H.T
        M = self.tanh(torch.matmul(self.w1,H))
        coef = self.sigmoid(self.coef)
        coef = 0
        alpha = self.softmax(torch.matmul(coef*self.w2+(1-coef)*tw2,M)).squeeze(2)
        r = self.tanh(torch.matmul(self.w3,torch.matmul(H.unsqueeze(0),alpha.unsqueeze(3))))
       # rplus = torch.cat([r,w2.permute(0,1,3,2).repeat(1,r.shape[1],1,1)],dim=2)
        logits = torch.matmul(self.w4,r)+self.b4
        logits = logits.permute(1,0,2,3)
        logits = logits.reshape(logits.shape[0],-1)
        return logits,w2

    def loss(self,logits,labels):
        logits = logits.reshape(logits.shape[0],-1,3).permute(0,2,1)
        loss = self.ce(logits,labels)
        return loss

    def judge(self,logits,labels):
        logits = logits.cpu()
        labels = labels.cpu().numpy().flatten()
        
        splitted = torch.split(logits,3,dim=1)
        logits = np.concatenate(list(map(lambda x:torch.argmax(x,dim=1,keepdim=True),splitted)),axis=1).flatten()
           
        validindex = (labels!=-1)
        return logits[validindex],labels[validindex]   


        '''
        aspect_correct = np.sum(labels==logits)
        aspect_num = len(labels) - np.sum(labels==-1)

        return aspect_correct,aspect_num
        '''

