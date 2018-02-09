import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import pretrainedmodels
#from torchvision import models
from torch.autograd import Variable
class AvgPool(nn.Module):
    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x, (x.size(2), x.size(3)))

class MaxPool(nn.Module):
    def forward(self, x):
        return torch.nn.functional.max_pool2d(x, (x.size(2), x.size(3)))

class DyiNet(nn.Module):
    def __init__(self, embedding_size,pool='max', num_classes=10, model_name='resnet152', pretrained=True, checkpoint=None):
        super().__init__()
        self.net = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        if pool=='max':
            self.net.avgpool = MaxPool()
        else:
            self.net.avgpool = AvgPool()
        
        self.fc = nn.Sequential(
            nn.Linear(self.net.last_linear.in_features+1, 1024),
            nn.Dropout(0.3),
            nn.Linear(1024,embedding_size),
            nn.Dropout(0.3),
            nn.Linear(128,num_classes),
        )
        self.centers = torch.zeros(num_classes, embedding_size).type(torch.FloatTensor)
        self.num_classes = num_classes
        self.classifier = nn.Linear(embedding_size, num_classes) 
        self.apply(self.weights_init)

        if checkpoint is not None:
            # Check if there are the same number of classes
            if list(checkpoint['state_dict'].values())[-1].size(0) == num_classes:
                print('weights loaded')
                self.load_state_dict(checkpoint['state_dict'])
                self.centers = checkpoint['centers']
            else:
                own_state = self.state_dict()
                for name, param in checkpoint['state_dict'].items():
                    if "classifier" not in name:
                        if isinstance(param, Parameter):
                            # backwards compatibility for serialized parameters
                            param = param.data
                        own_state[name].copy_(param)

    def weights_init(self,m):
        classname = m.__class__.__name__
#        if classname.find('Conv') != -1:
#            m.weight.data.normal_(0.0, 0.02)
#            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#            m.weight.data.normal_(0, math.sqrt(2. / n))
#            if m.bias is not None:
#                m.bias.data.zero_()
#        elif classname.find('BatchNorm') != -1:
#            m.weight.data.fill_(1)
#            m.bias.data.zero_()
        if classname.find('Linear') != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def get_center_loss(self,target, alpha):
        batch_size = target.size(0)
        features_dim = self.features.size(1)

        target_expand = target.view(batch_size,1).expand(batch_size,features_dim)

        centers_var = Variable(self.centers)
        centers_batch = centers_var.gather(0,target_expand.cpu()).cuda()

        criterion = nn.MSELoss()
        center_loss = criterion(self.features,  centers_batch)

        diff = centers_batch - self.features

        unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)

        appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))

        appear_times_expand = appear_times.view(-1,1).expand(batch_size,features_dim).type(torch.FloatTensor)

        diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)

        #∆c_j =(sum_i=1^m δ(yi = j)(c_j − x_i)) / (1 + sum_i=1^m δ(yi = j))
        diff_cpu = alpha * diff_cpu

        for i in range(batch_size):
            #Update the parameters c_j for each j by c^(t+1)_j = c^t_j − α · ∆c^t_j
            self.centers[target.data[i]] -= diff_cpu[i].type(self.centers.type())

        return center_loss, self.centers


    def fresh_params(self):
        return self.net.fc.parameters()
    
    def forward(self, x, O): #0, 1, 2, 3 -> (0, 3, 1, 2)
        out = torch.transpose(x, 1, 3) #0, 3, 2, 1
        out = torch.transpose(out, 2, 3) #0, 3, 1, 2
        #out = self.net(out)
        #out = self.net.features(x)
        out = self.net.features(out)
        out = self.net.avgpool(out)
        out = out.view(out.size(0), -1)
        out = torch.cat([out, O], 1)
        #print(out.shape)
        out = self.fc(out)
        self.features = out

        #out = self.classifier(out)
        return out
