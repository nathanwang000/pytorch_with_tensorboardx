import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# step 0: utilities
def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)    

# step 1: define your data
class Data(Dataset):
    def __init__(self, valid_percent=0.1, test_percent=0.2, random_seed=10, n=30000):
        super(self.__class__, self).__init__()        
        np.random.seed(random_seed)

        X = np.random.randn(n, 10)
        y = (X[:,1] > 0).astype(np.int)
        self.data = list(zip(X, y))

        test_start = int((1-test_percent) * len(self.data))
        val_start = int((1-(test_percent+valid_percent)) * len(self.data))
        self.train_ind = np.arange(0, val_start)
        self.val_ind = np.arange(val_start, test_start)
        self.test_ind = np.arange(test_start, len(self.data))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return (torch.from_numpy(np.array(x)).float(),
                torch.from_numpy(np.array([y])).long())


# step 2: define your network
class Net(nn.Module):
    def __init__(self, input_size=10, hidden_size=30, num_classes=2):
        super(self.__class__, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def embedding(self, x):
        res = []
        res.append(('input', x.data))
        # extract other features
        res.append(('fc1', self.relu(self.fc1(x)).data))
        return res

net = Net()
if torch.cuda.is_available():
    net.cuda()

# step 3: train the network
class Trainer:

    def __init__(self, data,
                 batch_size=256,
                 epoch_limit=1000,
                 early_stopping_limit=0,
                 criterion = nn.CrossEntropyLoss(),
                 save_loc='models/',
                 name="default",
                 log_dir=None):

        self.writer = SummaryWriter(log_dir=None)
        
        # hyper parameters
        self.epoch_limit = epoch_limit
        self.early_stopping_limit = early_stopping_limit
        self.save_loc = save_loc
        self.batch_size = batch_size
        self.criterion = criterion
        self.name = name

        # setup data
        self.data = data
        
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(data.train_ind)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(data.val_ind)
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(data.test_ind)
        retrain_sampler = torch.utils.data.sampler.SubsetRandomSampler(
            np.concatenate((data.train_ind, data.val_ind)))

        self.trainData = DataLoader(data,
                                    batch_size=batch_size, 
                                    sampler=train_sampler, 
                                    num_workers = 1)
        self.valData = DataLoader(data, 
                                  batch_size=batch_size, 
                                  sampler=val_sampler, 
                                  num_workers = 1)
        self.testData = DataLoader(data,
                                   batch_size=batch_size, 
                                   sampler=test_sampler, 
                                   num_workers = 1)
        self.retrainData = DataLoader(data,
                                      batch_size=batch_size, 
                                      sampler=retrain_sampler, 
                                      num_workers = 1)

    def train(self, model, optimizer=None):

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters())

        valid_loss = []
        train_loss = []
        retrain_loss = []

        bsf_val = np.inf
        bsf_ind = -1
        es = 0
        
        for i in range(self.epoch_limit):

            # validate for early stopping
            print('epoch {}'.format(i))
            running_valid_loss = 0
            for x, y in self.valData:
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                x, y = Variable(x).float(), Variable(y).view(-1).long()
                y_p = model(x)

            loss = self.criterion(y_p, y)
            running_valid_loss += to_np(loss)[0]
            self.writer.add_scalar('data/val_loss', loss.data[0], i) 

            val_loss = running_valid_loss/len(self.data.val_ind)*self.batch_size
            valid_loss.append(val_loss)

         
            if valid_loss[-1] < bsf_val and len(train_loss) > 0:
               
                print('Improved validation loss from {:.5f} to {:.5f}, train loss\
                {:.5f}'.format(bsf_val, valid_loss[-1],  train_loss[-1]))

                es = 0
                bsf_val = valid_loss[-1]
                bsf_ind = i
                self.es_best = i
                torch.save(model.state_dict(),
                           '{}_best.pt'.format(self.save_loc + self.name))
            elif len(train_loss) > 0:
                es += 1
                print('Validation loss {:.5f} does not improve best of {:.5f}, early stopping at {}/{}, train loss {:.5f}'.format(
                    valid_loss[-1], 
                    bsf_val, 
                    es, 
                    self.early_stopping_limit,
                    train_loss[-1]))
                if es > self.early_stopping_limit:
                    print('stopping')
                    break

            running_train_loss = 0
            for x, y in self.trainData:
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                x, y = Variable(x).float(), Variable(y).view(-1).long()
                y_p = model(x)
                optimizer.zero_grad()
                loss = self.criterion(y_p, y)
                running_train_loss += to_np(loss)[0]
                self.writer.add_scalar('data/train_loss', loss.data[0], i)   
                loss.backward()
                optimizer.step()
            train_loss.append(running_train_loss / len(self.data.train_ind) *
                              self.batch_size)

            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                self.writer.add_histogram(tag, to_np(value), i)
                self.writer.add_histogram(tag+'/grad', to_np(value.grad), i)    


        # plot first batch embedding of the train input
        for x, y in self.trainData:
            labels = list(map(str, y.numpy().ravel()))
            if torch.cuda.is_available():
                x = x.cuda()
            for step, (tag, em) in enumerate(model.embedding(Variable(x))):
                self.writer.add_embedding(em, metadata=labels, tag=tag, global_step=step)
            break
            

    def retrain(self, model, optimizer=None):

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters())

        for i in range(self.es_best):
            running_retrain_loss = 0
            for x, y in self.retrainData:
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                x, y = Variable(x).float(), Variable(y).view(-1).long()
                y_p = bn_val(x)
                optimizer.zero_grad()
                loss = self.criterion(y_p, y)
                running_retrain_loss += to_np(loss)[0]
                loss.backward()
                optimizer.step()
            retrain_loss.append(running_retrain_loss/ (len(self.data.train_ind)\
                                                       + len(self.data.val_ind)) * \
                                self.batch_size)
        torch.save(bn_val.state_dict(), '{}_full.pt'.format(save_loc+name))
        joblib.dump((train_loss, valid_loss, retrain_loss),
                    '{}_losses.pt'.format(save_loc+name))

t = Trainer(Data())
t.train(net)

