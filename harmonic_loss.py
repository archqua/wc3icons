import torch

def harmonic_loss(dist_fn, arg, im, n=4, sigma=0.015625, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    # weights = torch.zeros((arg.shape[0], arg.shape[1], n, n, n, n), dtype=torch.float32, device=device)
    # arg_dists = torch.zeros((arg.shape[0], arg.shape[1], n, n, n, n), dtype=torch.float32, device=device)
    # im_dists = torch.zeros((im.shape[0], im.shape[1], n, n, n, n), dtype=torch.float32, device=device)
    loss = torch.zeros((1,), dtype=torch.float32, device=device)
    for i in range(n):
        for j in range(n):
            for k in range(i, n):
                for l in range(j, n):
                    il, ir = (arg.shape[2] * i) // n, (arg.shape[2] * (i+1)) // n
                    jl, jr = (arg.shape[3] * j) // n, (arg.shape[3] * (j+1)) // n
                    kl, kr = (arg.shape[2] * k) // n, (arg.shape[2] * (k+1)) // n
                    ll, lr = (arg.shape[3] * l) // n, (arg.shape[3] * (l+1)) // n
                    # arg_dists[:, :, i, j, k, l] = dist_fn(arg[:, :, il:ir, jl:jr], arg[:, :, kl:kr, ll:lr])
                    # im_dists[:, :, i, j, k, l] = dist_fn(im[:, :, il:ir, jl:jr], im[:, :, kl:kr, ll:lr])
                    # weights[:, :, i, j, k, l] = torch.exp(torch.mul(arg_dists[:, :, i, j, k, l], -1/sigma))
                    arg_dists = dist_fn(arg[:, :, il:ir, jl:jr], arg[:, :, kl:kr, ll:lr])
                    im_dists  = dist_fn(im[:, :, il:ir, jl:jr], im[:, :, kl:kr, ll:lr])
                    weights   = torch.exp(-arg_dists / sigma)
                    loss += im_dists * weights
    return loss / (n**2)
    # return torch.mean(torch.mul(im_dists, weights))
