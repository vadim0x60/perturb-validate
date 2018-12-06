import torch
from torch import nn
from torch.distributions import Normal
import numpy as np

normal_dist = Normal(0, 1)

def multi_layer_perceptron(input_size, output_size, hidden_layers, activation=nn.Tanh):
    sizes = np.linspace(input_size, output_size, hidden_layers+2)
    sizes = sizes.round().astype(int)
    input_sizes = sizes[:-1]
    output_sizes = sizes[1:]

    layers = []
    for input_size, output_size in zip(input_sizes, output_sizes):
        layers.append(nn.Linear(input_size, output_size))
        layers.append(activation())
    return nn.Sequential(*layers)

class OrthodoxNet(nn.Module):
    def __init__(self, word_emb_size, hidden_size, out_layers):
        super().__init__()
        self.lstm = nn.LSTM(word_emb_size, hidden_size)
        self.classifier = multi_layer_perceptron(2 * hidden_size, 1, out_layers)

    def cuda(self):
        gpu_version = super().cuda()
        gpu_version.use_cuda = True
        return gpu_version

    def forward(self, sentences):
        c0 = normal_dist.sample((1, len(sentences), 32))
        h0 = normal_dist.sample((1, len(sentences), 32))
        

        if self.use_cuda:
            c0, h0, sentences = c0.cuda(), h0.cuda(), sentences.cuda()

        _, (cn, hn) = self.lstm(torch.Tensor(sentences), (c0, h0))
        return self.classifier(torch.cat((cn[0], hn[0]), dim=1))[:,0]

def validate_sentences(model, sentences):
    return model(torch.Tensor(sentences)).cpu().data.numpy().round().astype(int)
