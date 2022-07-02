import torch
from torch import nn

class DenseBlock(nn.Module):
    def __init__(
                self, width, length, activation=nn.ReLU(),
                kernel_size=3, stride=1, bias=False,
            ):
        super().__init__()

        self.activation = activation

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    width * i, width,
                    kernel_size=kernel_size, stride=stride, padding='same',
                    bias = (i == 1) and bias,
                ),
                nn.BatchNorm2d(width),
            ) for i in range(1, length + 1)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.hstack((x, self.activation(layer(x))))
        return x


class DensePix(nn.Module):
    # def __init__(self):
    #     super().__init__()

    class Transformator(nn.Module):
        def __init__(
                    self, width=16, dense_act=nn.ReLU(), conv_act=nn.LeakyReLU(0.01),
                    density=3, random_dim=128, dropout=0.5,
                ):
            super().__init__()

            self.activation = conv_act
            self.width = width
            self.random_dim = random_dim

            self.start = nn.Sequential(
                nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(width),
            )

            self.dense0 = DenseBlock(width, density, activation=dense_act)
            self.conv0 = nn.Sequential(
                nn.Conv2d(
                    density * width, 2 * width,
                    kernel_size=4, stride=2, padding=1, bias=False,
                ),
                nn.BatchNorm2d(2*width)
            ) # 56 -> 28

            self.dense1 = DenseBlock(2 * width, density, activation=dense_act)
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    density * 2 * width, 4 * width,
                    kernel_size=4, stride=2, padding=1 + 2, bias=False,
                ),
                nn.BatchNorm2d(4*width),
            ) # 28 -> 16

            self.dense2 = DenseBlock(4 * width, density, activation=dense_act)
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    density * 4 * width, 8 * width,
                    kernel_size=4, stride=2, padding=1, bias=False,
                ),
                nn.BatchNorm2d(8*width),
            ) # 16 -> 8

            self.dense3 = DenseBlock(8 * width, density, activation=dense_act)
            self.conv3 = nn.Sequential(
                nn.Conv2d(
                    density * 8 * width, 16 * width,
                    kernel_size=4, stride=2, padding=1, bias=False,
                ),
                nn.BatchNorm2d(16*width),
            ) # 8 -> 4

            self.dense4 = DenseBlock(16 * width, density, activation=dense_act)
            self.conv4 = nn.Sequential(
                nn.Conv2d(
                    density * 16 * width, 32 * width,
                    kernel_size=3, stride=1, padding=0, bias=False,
                ),
                nn.BatchNorm2d(32*width),
            ) # 4 -> 2

            self.dense5 = DenseBlock(32 * width, density, activation=dense_act)
            self.conv5 = nn.Sequential(
                nn.Conv2d(
                    density * 32 * width, 64 * width,
                    kernel_size=2, stride=1, padding=0, bias=False,
                ),
                # nn.BatchNorm2d(64 * width), # dropout
            ) # 2 -> 1

            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(64 * width + random_dim, 64 * width + random_dim),
                nn.Unflatten(dim=1, unflattened_size=(64  * width + random_dim, 1, 1)),
            )

            if density > 1:
                density -= 1

            self.ups0 = nn.Sequential(
                nn.ConvTranspose2d(
                    64 * width + random_dim, 32 * width,
                    kernel_size=2, stride=1, padding=0, bias=True,
                ),
                nn.BatchNorm2d(32 * width),
            ) # 1 -> 2
            self.esned0 = DenseBlock(32 * width, density, activation=dense_act)

            self.ups1 = nn.Sequential(
                nn.ConvTranspose2d(
                    (density+2) * 32 * width, 16 * width,
                    kernel_size=3, stride=1, padding=0, bias=False,
                ),
                nn.BatchNorm2d(16 * width),
            ) # 2 -> 4
            self.esned1 = DenseBlock(16 * width, density, activation=dense_act)

            self.ups2 = nn.Sequential(
                nn.ConvTranspose2d(
                    (density+2) * 16 * width, 8 * width,
                    kernel_size=4, stride=2, padding=1, bias=False,
                ),
                nn.BatchNorm2d(8 * width),
            ) # 4 -> 8
            self.esned2 = DenseBlock(8 * width, density, activation=dense_act)

            self.ups3 = nn.Sequential(
                nn.ConvTranspose2d(
                    (density+2) * 8 * width, 4 * width,
                    kernel_size=4, stride=2, padding=1, bias=False,
                ),
                nn.BatchNorm2d(4 * width),
            ) # 8 -> 16
            self.esned3 = DenseBlock(4 * width, density, activation=dense_act)

            self.ups4 = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2), # 16 -> 32
                nn.Conv2d(
                    (density+2) * 4 * width, 2 * width,
                    kernel_size=5, stride=1, padding=0, bias=False,
                ), # 32 -> 28
                nn.BatchNorm2d(2 * width),
            )
            self.esned4 = DenseBlock(2 * width, density, activation=dense_act)

            self.ups5 = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(
                    (density+2) * 2 * width, width,
                    kernel_size=3, stride=1, padding=1, bias=False,
                ),
                nn.BatchNorm2d(width),
            )
            self.esned5 = DenseBlock(width, density, activation=dense_act)

            self.term = nn.Sequential(
                nn.Conv2d(
                    (density+2) * width, 3,
                    kernel_size=3, stride=1, padding=1, bias=False,
                ),
                # nn.Sigmoid(), # in forward
            )

        def forward(self, x, random_component=None, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
            x = self.activation(self.start(x))
            ftrs = []

            for i in range(6):
                width = self.width * 2**(i)
                # print(x.shape)
                x = eval(f"self.activation(self.dense{i}(x))")
                # print(x.shape)
                ftrs.append(x[:, -width:])
                x = eval(f"self.activation(self.conv{i}(x[:, :-width]))")

            if self.random_dim > 0:
                if random_component is None:
                    rnd = torch.randn(size=(x.shape[0], self.random_dim, 1, 1), device=device, dtype=torch.float32)
                else:
                    rnd = random_component
                x = torch.hstack((x, rnd))
            x = self.fc(x)

            for i in range(6):
                if i > 0:
                    x = torch.hstack((x, ftrs[-i]))
                x = eval(f"self.activation(self.ups{i}(x))")
                x = eval(f"self.activation(self.esned{i}(x))")

            x = torch.hstack((x, ftrs[0]))
            x = self.term(x)
            return torch.sigmoid(x)
