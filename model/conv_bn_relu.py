import torch.nn as nn

class ConvBNRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, padding=0)
        #self.batch_norm = nn.BatchNorm2d(channels_out)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.00000001)  # Small slope for minimal change

    def forward(self, x):
        x = self.conv(x)
        #x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return x

# class ConvBNRelu(nn.Module):
#     def __init__(self, channels_in, channels_out, stride=1):
#         super(ConvBNRelu, self).__init__()
#         self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, padding=0)
#         self.norm = nn.InstanceNorm2d(channels_out, affine=True)  # Replacing BatchNorm2d with InstanceNorm2d
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.norm(x)
#         x = self.leaky_relu(x)
#         return x