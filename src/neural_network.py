from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding=0):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, padding=padding)
    # We don't need the bias when used with BN because the BN also contains a bias
    self.bn = nn.BatchNorm2d(out_channels)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.pool(x)
    return F.relu(x, inplace=True)


class MLBioNN(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.conv1 = ConvBlock(3, 10, 3, padding=1)     # 3 * 450 * 600 * 10
        self.conv2 = ConvBlock(10, 20, 3, padding=1)    # 10 * 225 * 300 * 20
        self.conv3 = ConvBlock(20, 40, 3, padding=1)    # 20 * 112 * 150 * 40
        self.conv4 = ConvBlock(40, 80, 3, padding=1)    # 40 * 56 * 75 * 80
        self.flatten = nn.Flatten()                     # 80 * 28 * 37
        self.dense = nn.Linear(80 * 28 * 37, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)

        return self.dense(x)

