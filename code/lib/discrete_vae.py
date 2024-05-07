import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from math import sqrt

def exists(val):
    return val is not None

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def default(val, d):
    return val if exists(val) else d

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -torch.log(-torch.log(noise))

def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        return self.net(x) + x

class DiscreteVAE(nn.Module):
    def __init__(
        self,
        image_size=224,
        num_tokens=512,
        codebook_dim=512,
        num_layers=3,
        num_resnet_blocks=2,
        hidden_dim=128,
        channels=3,
        smooth_l1_loss=False,
        temperature=0.9,
        straight_through=False,
        reinmax=False,
        kl_div_loss_weight=0.,
        normalization=((0.5,) * 3, (0.5,) * 3)
    ):
        super().__init__()
        assert num_layers >= 1, 'Number of layers must be >= 1'
        has_resblocks = num_resnet_blocks > 0

        self.channels = channels
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.temperature = temperature
        self.straight_through = straight_through
        self.reinmax = reinmax

        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        enc_channels = [channels, *[hidden_dim] * num_layers]
        dec_channels = [codebook_dim if not has_resblocks else hidden_dim, *reversed(enc_channels[1:])]

        enc_layers = []
        dec_layers = []

        for enc_in, enc_out in zip(enc_channels[:-1], enc_channels[1:]):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, enc_out, 4, stride=2, padding=1), nn.ReLU()))
        
        for dec_in, dec_out in zip(dec_channels[:-1], dec_channels[1:]):
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, dec_out, 4, stride=2, padding=1), nn.ReLU()))

        if has_resblocks:
            for _ in range(num_resnet_blocks):
                enc_layers.append(ResBlock(enc_channels[-1]))
                dec_layers.insert(0, ResBlock(hidden_dim))
            dec_layers.insert(0, nn.Conv2d(codebook_dim, hidden_dim, 1))

        enc_layers.append(nn.Conv2d(enc_channels[-1], num_tokens, 1))
        dec_layers.append(nn.Conv2d(dec_channels[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

        self.normalization = tuple(map(lambda t: t[:channels], normalization))

    def norm(self, images):
        if not exists(self.normalization):
            return images
        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c () ()'), (means, stds))
        return (images.clone().sub_(means).div_(stds))

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self(images, return_logits=True)
        return logits.argmax(dim=1).flatten(1)

    def denormalize(self, images):
        device = images.device
        means, stds = map(lambda t: torch.tensor(t, device=device).reshape(1, 3, 1, 1), ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        return images * stds + means

    def decode(self, img_seq):
        image_embeds = rearrange(self.codebook(img_seq), 'b (h w) d -> b d h w', h=int(sqrt(img_seq.shape[1])))
        return self.decoder(image_embeds)

    def forward(self, img, return_loss=True, return_recons=True, return_logits=False, temp=None):
        img = self.norm(img)
        logits = self.encoder(img)

        if return_logits:
            return logits

        temp = default(temp, self.temperature)
        one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=self.straight_through)
        
        if self.straight_through and self.reinmax:
            one_hot = one_hot.detach()
            pi_0 = logits.softmax(dim=1)
            pi_1 = (one_hot + (logits / temp).softmax(dim=1)) / 2
            pi_1 = ((log(pi_1) - logits).detach() + logits).softmax(dim=1)
            one_hot = 2 * pi_1 - 0.5 * pi_0 - (2 * pi_1 - 0.5 * pi_0).detach() + one_hot

        sampled = einsum('b n h w, n d -> b d h w', one_hot, self.codebook.weight)
        out = self.denormalize(self.decoder(sampled))

        if not return_loss:
            return out

        recon_loss = self.loss_fn(img, out)
        log_qy = F.log_softmax(rearrange(logits, 'b n h w -> b (h w) n'), dim=-1)
        kl_div = F.kl_div(torch.log(torch.tensor([1. / self.num_tokens], device=img.device)), log_qy, None, None, 'batchmean', log_target=True)
        
        if not return_recons:
            return recon_loss + kl_div * self.kl_div_loss_weight

        return recon_loss + kl_div * self.kl_div_loss_weight, out
