import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()

        self.conv1 = nn.Conv1d(40, 32, kernel_size=3, stride=1, padding=1)
        #### START CODE: ADD NEW LAYERS ####
        # (do not forget to update `flattened_size`:
        # the input size of the first fully connected layer self.fc1)
        # self.conv2 = ...
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1)

        # Size of the output of the last convolution:
        self.flattened_size = 3232
        ### END CODE ###

        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.fc3 = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass,
        x shape is (batch_size, 3, 32, 32)
        (color channel first)
        in the comments, we omit the batch_size in the shape
        """

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        # Check the output size
        output_size = np.prod(x.size()[1:])
        assert output_size == self.flattened_size,\
                "self.flattened_size is invalid {} != {}".format(output_size, self.flattened_size)

        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)

        return x



class CNNTradPool2(nn.Module):
    def __init__(self):
        super(CNNTradPool2, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(20, 8), stride=(1, 1))
        self.pool1 = nn.MaxPool2d((2, 2))

        x = Variable(torch.zeros(1, 1, 40, 101), volatile=True)
        x = self.pool1(self.conv1(x))
        #### START CODE: ADD NEW LAYERS ####
        # (do not forget to update `flattened_size`:
        # the input size of the first fully connected layer self.fc1)
        # self.conv2 = ...
        self.conv2 = nn.Conv1d(64, 64, kernel_size=(10, 4), stride=(1, 1))
        self.pool2 = nn.MaxPool2d((1, 1))

        x = self.pool2(self.conv2(x))
        conv_net_size = x.view(1, -1).size(1)

        self.output = nn.Linear(conv_net_size, 1)
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Forward pass,
        x shape is (batch_size, 3, 32, 32)
        (color channel first)
        in the comments, we omit the batch_size in the shape
        """

        x = F.relu(self.conv1(x.unsqueeze(1)))
        x = self.dropout(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        #x = F.relu(self.conv3(x))
        # Check the output size
        # output_size = np.prod(x.size()[1:])
        # assert output_size == self.flattened_size,\
        #        "self.flattened_size is invalid {} != {}".format(output_size, self.flattened_size)

        # x = x.view(-1, self.flattened_size)
        x = self.output(x)
        x = self.sig(x)
        return x
