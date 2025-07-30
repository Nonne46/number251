import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()
        self.time_mlp = nn.Linear(time_dim, out_channels)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # Add time embedding
        time_emb = self.time_mlp(self.act(t))
        h = h + time_emb[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Conv2d(channels, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)

        qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        q, k, v = map(
            lambda t: t.view(b, self.heads, self.dim_head, h * w).transpose(-1, -2), qkv
        )

        attn = torch.matmul(q, k.transpose(-1, -2)) * (self.dim_head**-0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(-1, -2).reshape(b, self.heads * self.dim_head, h, w)
        return self.to_out(out) + x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, use_attn=False):
        super().__init__()
        self.conv_block1 = ConvBlock(in_channels, out_channels, time_dim)
        self.conv_block2 = ConvBlock(out_channels, out_channels, time_dim)
        self.attn = AttentionBlock(out_channels) if use_attn else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x, t):
        x = self.conv_block1(x, t)
        x = self.conv_block2(x, t)
        x = self.attn(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(
        self, in_channels, skip_channels, out_channels, time_dim, use_attn=False
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, 3, stride=2, padding=1, output_padding=1
        )
        self.conv_block1 = ConvBlock(
            in_channels + skip_channels, out_channels, time_dim
        )
        self.conv_block2 = ConvBlock(out_channels, out_channels, time_dim)
        self.attn = AttentionBlock(out_channels) if use_attn else nn.Identity()

    def forward(self, x, skip, t):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block1(x, t)
        x = self.conv_block2(x, t)
        x = self.attn(x)
        return x


class SS13MapDiffusion(nn.Module):
    def __init__(self, vocab_size, layers=16, base_channels=64, time_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.layers = layers

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, base_channels)

        # Initial projection
        self.init_conv = nn.Conv2d(layers * base_channels, base_channels, 3, padding=1)

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # U-Net encoder
        self.down1 = DownBlock(base_channels, base_channels * 2, time_dim)
        self.down2 = DownBlock(
            base_channels * 2, base_channels * 4, time_dim, use_attn=True
        )
        self.down3 = DownBlock(
            base_channels * 4, base_channels * 8, time_dim, use_attn=True
        )

        # Bottleneck
        self.mid_block1 = ConvBlock(base_channels * 8, base_channels * 8, time_dim)
        self.mid_attn = AttentionBlock(base_channels * 8)
        self.mid_block2 = ConvBlock(base_channels * 8, base_channels * 8, time_dim)

        # U-Net decoder
        self.up3 = UpBlock(
            base_channels * 8,
            base_channels * 8,
            base_channels * 4,
            time_dim,
            use_attn=True,
        )
        self.up2 = UpBlock(
            base_channels * 4,
            base_channels * 4,
            base_channels * 2,
            time_dim,
            use_attn=True,
        )
        self.up1 = UpBlock(
            base_channels * 2, base_channels * 2, base_channels, time_dim
        )

        # Output projection
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, layers * vocab_size, 3, padding=1)

    def forward(self, x, t):
        # x shape: (batch, layers, height, width)
        batch, layers, h, w = x.shape

        # Embed tokens
        x_embed = self.token_embed(x)  # (batch, layers, h, w, channels)
        x_embed = x_embed.permute(0, 1, 4, 2, 3)  # (batch, layers, channels, h, w)
        x_embed = x_embed.reshape(batch, -1, h, w)  # (batch, layers*channels, h, w)

        # Initial conv
        x = self.init_conv(x_embed)

        # Time embedding
        t_emb = self.time_mlp(t)

        # Encoder
        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        x, skip3 = self.down3(x, t_emb)

        # Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # Decoder
        x = self.up3(x, skip3, t_emb)
        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)

        # Output
        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)

        # Reshape to (batch, layers, vocab_size, h, w)
        x = x.reshape(batch, layers, self.vocab_size, h, w)

        return x


class SS13MapDiffusionLightning(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        layers=16,
        base_channels=64,
        timesteps=1000,
        learning_rate=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = SS13MapDiffusion(vocab_size, layers, base_channels, 512)
        self.timesteps = timesteps
        self.vocab_size = vocab_size

        # Noise schedule (linear for discrete diffusion)
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, timesteps))
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process - corrupt data"""
        if noise is None:
            # For discrete diffusion, we randomly replace tokens
            noise = torch.randint(0, self.vocab_size, x_0.shape, device=x_0.device)

        # Get alpha values for timestep t
        alpha_t = self.alphas_cumprod[t][:, None, None, None]

        # Probability of keeping original token
        keep_mask = torch.rand_like(x_0, dtype=torch.float32) < alpha_t

        # Mix original and noise
        x_t = torch.where(keep_mask, x_0, noise)

        return x_t, noise

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        # Reshape from dataset format
        x = batch["tensor_data"]  # (batch, layers, h, w)
        x = torch.tensor(x, dtype=torch.long, device=self.device)

        # Sample timesteps
        batch_size = x.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)

        # Add noise
        x_noisy, noise = self.q_sample(x, t)

        # Predict noise
        logits = self(x_noisy, t)  # (batch, layers, vocab_size, h, w)

        # Calculate loss
        loss = F.cross_entropy(
            logits.transpose(2, 1),  # (batch, vocab_size, layers, h, w)
            x,  # (batch, layers, h, w)
            reduction="mean",
        )

        # class_counts = torch.bincount(x.flatten(), minlength=self.vocab_size)
        # class_weights = 1.0 / torch.log(
        #     1.2 + class_counts.float()
        # )  # Logarithmic weighting
        # class_weights = class_weights / class_weights.sum() * self.vocab_size
        #
        # weight_loss = F.cross_entropy(
        #     logits.view(-1, self.vocab_size),  # Flatten all dimensions except class
        #     x.view(-1),  # Flatten target
        #     weight=class_weights.to(self.device),
        #     reduction="mean",
        # )
        #
        # loss += weight_loss * 0.3

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["tensor_data"]
        x = torch.tensor(x, dtype=torch.long, device=self.device)

        batch_size = x.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)

        x_noisy, noise = self.q_sample(x, t)
        logits = self(x_noisy, t)

        loss = F.cross_entropy(logits.transpose(2, 1), x, reduction="mean")

        # class_counts = torch.bincount(x.flatten(), minlength=self.vocab_size)
        # class_weights = 1.0 / torch.log(
        #     1.2 + class_counts.float()
        # )  # Logarithmic weighting
        # class_weights = class_weights / class_weights.sum() * self.vocab_size
        #
        # weight_loss = F.cross_entropy(
        #     logits.view(-1, self.vocab_size),  # Flatten all dimensions except class
        #     x.view(-1),  # Flatten target
        #     weight=class_weights.to(self.device),
        #     reduction="mean",
        # )
        #
        # loss += weight_loss * 0.3

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]

    @torch.no_grad()
    def sample(self, shape, device):
        """Generate new maps using DDPM sampling"""
        batch_size, layers, h, w = shape

        # Start from pure noise
        x = torch.randint(0, self.vocab_size, (batch_size, layers, h, w), device=device)

        # Denoise step by step
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device)

            # Predict denoised tokens
            logits = self(x, t_batch)
            probs = F.softmax(logits, dim=2)

            # Sample from distribution
            x_pred = torch.multinomial(
                probs.permute(0, 1, 3, 4, 2).reshape(-1, self.vocab_size), num_samples=1
            ).reshape(batch_size, layers, h, w)

            # Mix with less noise for next step
            if t > 0:
                alpha_t = self.alphas_cumprod[t]
                alpha_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)

                # Probability of replacing with predicted vs keeping noisy
                replace_prob = (alpha_t - alpha_prev) / alpha_t
                replace_mask = torch.rand_like(x, dtype=torch.float32) < replace_prob

                x = torch.where(replace_mask, x_pred, x)
            else:
                x = x_pred

        return x
