import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,RandomSampler,TensorDataset
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random
from model import JointModel
from sklearn.metrics import accuracy_score,f1_score

batch_size = 16
epochs = 20   
lr_small = 2e-5
lr_big =   1e-4       
device = 'cuda:0'
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer_path = 'KGRL/KGRL-main/pretrained'
model_path = 'KGRL/KGRL-main/pretrained'

save_path = 'checkpointmodel/model.pt'
record_path = 'recordcheck.txt'

train_file = 'KGRL/KGRL-main/processed/res_train.npy'
dev_file = 'KGRL/KGRL-main/processed/res_test.npy'

#fix random seed
def fixseed(seed=31):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

fixseed()

print('Loading dataset......')
train_data = np.load(train_file,allow_pickle=True)
dev_data = np.load(dev_file,allow_pickle=True)
assert(train_data.shape[1]%3==1)
assert(dev_data.shape[1]%3==1)
train_sen,train_label = train_data[:,0],torch.LongTensor(train_data[:,1:].astype('int8'))
dev_sen,dev_label = dev_data[:,0],torch.LongTensor(dev_data[:,1:].astype('int8'))
aspects_num = train_label.shape[1]
print('Aspect number:{}'.format(aspects_num))

print('Loading model...')
model = JointModel(model_path, tokenizer_path, aspects_num, ignore_label=-1,device=device)
#
model = model.to(device)
model = nn.DataParallel(model)
#model = nn.DataParallel(model).to(device)


print('building dataset...')
train_ids,train_masks = model.module.tokenize(train_sen)
dev_ids,dev_masks = model.module.tokenize(dev_sen)
train_dataset = TensorDataset(train_ids,train_masks,train_label)
dev_dataset = TensorDataset(dev_ids,dev_masks,dev_label)
train_dataloader = DataLoader(train_dataset,sampler=RandomSampler(train_dataset),batch_size=batch_size)
dev_dataloader = DataLoader(dev_dataset,batch_size=batch_size)

print('batch numbers: train({}),test({})batchsize({})'.format(
    len(train_dataloader),len(dev_dataloader),batch_size))

print('config optimizer...')
optimizer = model.module.get_optimizer(lr_small,lr_big)

total_steps = len(train_dataloader)*epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,num_warmup_steps=0,num_training_steps=total_steps)

model.cuda()

training_stats = []
total_t0 = time.time()
for epoch_i in range(epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_train_loss = 0
    model.train()
    for step,batch in enumerate(train_dataloader):
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()
        b_labels = batch[2].cuda()
        
        model.zero_grad()
    
        logits,w2 = model(b_input_ids,b_input_mask)

        loss = model.module.loss(logits,b_labels)
        total_train_loss += loss.item()

        if step%2 == 0 and not step == 0:
            elapsed = format_time(time.time()-t0)
            print('Batch {:>5,} of {:>5,}.  Elapsed: {:}. batchloss:{:.4f}'.format(step,len(train_dataloader),elapsed,loss.item()))

        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        
        optimizer.step()
        scheduler.step()


    avg_train_loss = total_train_loss/len(train_dataloader)
    training_time = format_time(time.time()-t0)
    print("")
    print(" Average training loss: {0:.2f}".format(avg_train_loss))
    print(" Training epcoh took: {:}".format(training_time))
    

    # record epoch info
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Training Time': training_time,
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
print("")
print("Running Test...")
w2 = w2.to('cpu')
t0 = time.time()
model.eval()

total_eval_loss = 0
    
validpreds = []
validlabels = []

for batch in dev_dataloader:
    b_input_ids = batch[0].cuda()
    b_input_mask = batch[1].cuda()
    b_labels = batch[2].cuda()
    with torch.no_grad():        
        logits,_ = model(b_input_ids,b_input_mask,w2=w2.to(b_input_ids.device))
        loss = model.module.loss(logits,b_labels)
        
    total_eval_loss += loss.item()
        
    validpred,validlabel = model.module.judge(logits,b_labels)
    validpreds += list(validpred)
    validlabels += list(validlabel)

macro_f1 = f1_score(validlabels,validpreds,average='macro')
avg_aspect_accuracy = accuracy_score(validlabels,validpreds)
avg_val_loss = total_eval_loss / len(dev_dataloader)

print(" Aspect Acc: {0:.5f}"
            .format(avg_aspect_accuracy))
print("f1score:{0:.5f}".format(macro_f1))
validation_time = format_time(time.time() - t0)
    
print("  Validation Loss: {0:.4f}".format(avg_val_loss))
print("  Validation took: {:}".format(validation_time))

fileob = open(record_path,'w')
fileob.write(str(training_stats))
fileob.close()
##################
print("")
print('saving model...')
model.module.save_pretrained(save_path)
print('saving done')
