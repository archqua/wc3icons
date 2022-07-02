import torch
from torch import nn

class Pix(nn.Module):
    def __init__(
                self,
                g_width=16, g_latent=64, g_random=None, g_act=nn.ReLU(), g_drop=0.5,
                d_width=16, d_act=nn.LeakyReLU(0.1), d_drop=0.5,
                crop=4,
            ):
        super().__init__()

        self.trans = Pix.Transformator(g_width, g_latent, g_random, g_act, g_drop)
        self.disc = Pix.Discriminator(d_width, d_act, d_drop)
    

    class Transformator(nn.Module):
        def __init__(self, width=16, latent_dim=64, random_dim=None, activation=nn.ReLU(), dropout=0.5, crop=4):
            super().__init__()

            self.crop = crop
            self.latent_dim = latent_dim
            self.random_dim = random_dim
            if random_dim is not None:
                self.delatent_dim = latent_dim + random_dim
            else:
                self.delatent_dim = latent_dim
            self.activation = activation
            
            self.encoder = nn.ModuleList()
            self.encoder.append(nn.Sequential(
                nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(width),
            )) # 56 -> 56
            for i in range(4): # 56 -> 4
                self.encoder.append(nn.Sequential(
                    nn.Conv2d(
                        width * 2**i, width * 2**(i+1),
                        kernel_size=4, stride=2, padding=1 + (crop//2)*((i==1)), bias=False,
                    ),
                    nn.BatchNorm2d(width * 2**(i+1)),
                ))
            i = 4
            self.encoder.append(nn.Sequential(
                nn.Conv2d(width * 2**i, width * 2**(i+1), kernel_size=3, bias=False),
                nn.BatchNorm2d(width * 2**(i+1)),
            )) # 4 -> 2
            i = 5
            self.encoder.append(nn.Sequential(
                nn.Conv2d(width * 2**i, latent_dim, kernel_size=2),
                # nn.BatchNorm2d(latent_dim), # dropout
            )) # 2 -> 1

            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=dropout),
                nn.Linear(self.delatent_dim, self.delatent_dim, bias=True),
                # nn.BatchNorm1d(self.delatent_dim), # ruins everything??
                nn.Unflatten(dim=1, unflattened_size=(self.delatent_dim, 1, 1)),
            )
            self.decoder = nn.ModuleList()
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(self.delatent_dim, width * 2**i, kernel_size=2, bias=True),
                nn.BatchNorm2d(width * 2**i),
            )) # 1 -> 2
            i = 4
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(2 * width * 2**(i+1), width * 2**i, kernel_size=3, bias=False),
                nn.BatchNorm2d(width * 2**i),
            )) # 2 -> 4
            for i in range(3, 1, -1): # 4 -> 16
                self.decoder.append(nn.Sequential(
                    nn.ConvTranspose2d(
                        2 * width * 2**(i+1), width * 2**i,
                        kernel_size=4, stride=2, padding=1 + (crop//2)*((i==1)), bias=False,
                    ),
                    nn.BatchNorm2d(width * 2**i),
                ))
            i = 1
            self.decoder.append(nn.Sequential(
                nn.Conv2d(
                    2 * width * 2**(i+1), width * 2**(i),
                    kernel_size=3, stride=1, padding=0, bias=False,
                ), # 16 -> 14
                nn.BatchNorm2d(width * 2**(i)),
                # nn.PixelShuffle(2), # 14 -> 28
                nn.UpsamplingBilinear2d(scale_factor=2), # 14 -> 28
            ))
            i = 0
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(
                    2 * width * 2**(i+1), width * 2**(i),
                    kernel_size=3, stride=1, padding=1, bias=False,
                ),
                nn.BatchNorm2d(width * 2**i),
                nn.UpsamplingBilinear2d(scale_factor=2), # 28 -> 56
            ))
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(
                    2 * width, 3,
                    kernel_size=3, stride=1, padding=1, bias=False,
                ),
                # nn.Sigmoid(), # in forward
            )) # 64 -> 64

        def forward(self, x, random_component=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
            feature_maps = []
            for i in range(len(self.encoder) - 1):
                layer = self.encoder[i]
                x = self.activation(layer(x))
                # feature_maps.append(x.clone())
                feature_maps.append(x)
            x = self.activation(self.encoder[-1](x))

            if self.random_dim is not None:
                if random_component is None:
                    rnd = torch.randn(size=(x.shape[0], self.random_dim, 1, 1), device=device, dtype=torch.float32)
                else:
                    rnd = random_component
                x = torch.hstack((x, rnd))

            x = self.fc(x)

            for i in range(len(self.decoder) - 1):
                layer = self.decoder[i]
                x = self.activation(layer(x))
                # print(x.shape, feature_maps[-i-1].shape)
                x = torch.hstack((x, feature_maps[-i-1]))
            x = self.decoder[-1](x) # no activation here

            return torch.sigmoid(x)

    
    class Discriminator(nn.Module):
        def __init__(self, width=16, activation=nn.LeakyReLU(0.1), dropout=0.5, crop=4):
            super().__init__()

            self.crop = crop
            self.n_features = [3] + [width * 2**i for i in range(7)]
            self.kernel_sizes = [4] * 4 + [3] + [2]
            self.strides = [2] * 4 + [1] * 2
            self.paddings = [1] + [1 + crop//2] + [1] * 2 + [0] * 2
            self.activation = activation

            self.conv = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        self.n_features[i], self.n_features[i+1],
                        kernel_size = self.kernel_sizes[i],
                        stride = self.strides[i],
                        padding = self.paddings[i],
                        bias = (i == 0),
                    ),
                    nn.BatchNorm2d(self.n_features[i+1]),
                ) for i in range(5)
            ]) # 64 -> 2
            i = 5
            self.conv.append(nn.Sequential(
                nn.Conv2d(
                    self.n_features[i], self.n_features[i+1],
                    kernel_size = self.kernel_sizes[i],
                    stride = self.strides[i],
                    padding = self.paddings[i],
                    bias = False,
                ),
                # nn.BatchNorm2d(self.n_features[i+1]), # dropout
                nn.Flatten(),
            ))
            
            i = 6
            self.fc0 = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.n_features[i], self.n_features[i], bias=True),
                # nn.BatchNorm1d(self.n_features[i]), # breaks the whole thing
            )
            self.fc1 = nn.Linear(self.n_features[i], 1, bias=False)
            # sigmoid in forward

            self.counter = 0

        def represent(self, x):
            for layer in self.conv:
                x = self.activation(layer(x))
            return x

        def forward(self, x):
            x = self.represent(x)
            x = self.fc0(x)
            x = self.fc1(x)
            return torch.sigmoid(x)



