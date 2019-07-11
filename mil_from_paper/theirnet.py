from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F

#Implementation of Attention-MIL model in PyTorch

class TheirNet(nn.Module):
    def __init__(self):
        super(TheirNet, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(p = 0.6),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        pass
    
    def forward(self, x):
        #x is Nx512
        H = x

#         H = H.view(-1, 50 * 4 * 4)
#         H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # Nx1
        A = torch.transpose(A, 1, 0)  # 1xN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # 1x512

        Y_prob = self.classifier(M) #scalar
        Y_hat = torch.ge(Y_prob, 0.5).float() #class label
        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        
        return neg_log_likelihood, A
