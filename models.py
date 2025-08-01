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
    def __init__(self, in_channels, out_channels, time_dim, groups=8, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.time_mlp = nn.Linear(time_dim, out_channels)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.dropout(h)

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

        self.norm = nn.GroupNorm(min(8, channels), channels)
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

        # Token embedding with proper initialization
        self.token_embed = nn.Embedding(vocab_size, base_channels)
        nn.init.normal_(self.token_embed.weight, std=0.02)

        # Initial projection
        self.init_conv = nn.Conv2d(layers * base_channels, base_channels, 3, padding=1)

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # U-Net encoder - reduced complexity
        self.down1 = DownBlock(base_channels, base_channels, time_dim)
        self.down2 = DownBlock(
            base_channels, base_channels * 2, time_dim, use_attn=True
        )
        self.down3 = DownBlock(
            base_channels * 2, base_channels * 4, time_dim, use_attn=True
        )

        # Bottleneck
        self.mid_block1 = ConvBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid_attn = AttentionBlock(base_channels * 4)
        self.mid_block2 = ConvBlock(base_channels * 4, base_channels * 4, time_dim)

        # U-Net decoder
        self.up3 = UpBlock(
            base_channels * 4,
            base_channels * 4,
            base_channels * 2,
            time_dim,
            use_attn=True,
        )
        self.up2 = UpBlock(
            base_channels * 2, base_channels * 2, base_channels, time_dim, use_attn=True
        )
        self.up1 = UpBlock(base_channels, base_channels, base_channels, time_dim)

        # Output projection
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, layers * vocab_size, 3, padding=1)

        # Initialize output layer to zero for stable training
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

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
        timesteps=250,  # Reduced timesteps
        learning_rate=2e-4,
        mask_prob_min=0.1,
        mask_prob_max=0.9,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = SS13MapDiffusion(vocab_size, layers, base_channels, 512)
        self.timesteps = timesteps
        self.vocab_size = vocab_size
        self.mask_prob_min = mask_prob_min
        self.mask_prob_max = mask_prob_max

        # Linear noise schedule for discrete case
        betas = torch.linspace(mask_prob_min, mask_prob_max, timesteps)
        self.register_buffer("betas", betas)

        # For discrete diffusion, we use mask probabilities directly
        self.register_buffer("mask_probs", betas)

    def q_sample(self, x_0, t):
        """Forward diffusion process - mask tokens with probability based on timestep"""
        batch_size = x_0.shape[0]

        # Get mask probability for each sample based on timestep
        mask_prob = self.mask_probs[t]  # Shape: (batch_size,)

        # Create random mask for each position
        mask = (
            torch.rand_like(x_0, dtype=torch.float32) < mask_prob[:, None, None, None]
        )

        # Create random replacement tokens
        noise_tokens = torch.randint(0, self.vocab_size, x_0.shape, device=x_0.device)

        # Apply mask
        x_t = torch.where(mask, noise_tokens, x_0)

        return x_t, mask

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        x = batch["tensor_data"]
        x = torch.tensor(x, dtype=torch.long, device=self.device)

        batch_size = x.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)

        # Add noise
        x_noisy, mask = self.q_sample(x, t)

        # Predict original tokens
        logits = self(x_noisy, t)  # (batch, layers, vocab_size, h, w)

        # Reshape for loss computation
        logits = logits.permute(0, 1, 3, 4, 2)  # (batch, layers, h, w, vocab_size)
        logits = logits.reshape(-1, self.vocab_size)
        targets = x.reshape(-1)

        # Calculate base loss
        base_loss = F.cross_entropy(logits, targets, reduction="mean")

        class_counts = torch.bincount(targets, minlength=self.vocab_size)
        # Use square root instead of log for gentler weighting
        class_weights = 1.0 / torch.sqrt(1.0 + class_counts.float())
        class_weights = class_weights / class_weights.mean()  # Normalize to mean=1
        # Cap the maximum weight to prevent extreme reweighting
        class_weights = torch.clamp(class_weights, min=0.1, max=3.0)

        weighted_loss = F.cross_entropy(
            logits, targets, weight=class_weights.to(self.device), reduction="mean"
        )

        # Mix losses with small weight for balance
        loss = 0.8 * base_loss + 0.2 * weighted_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("base_loss", base_loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["tensor_data"]
        x = torch.tensor(x, dtype=torch.long, device=self.device)

        batch_size = x.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)

        x_noisy, mask = self.q_sample(x, t)
        logits = self(x_noisy, t)

        # Reshape for loss computation
        logits = logits.permute(0, 1, 3, 4, 2)
        logits = logits.reshape(-1, self.vocab_size)
        targets = x.reshape(-1)

        loss = F.cross_entropy(logits, targets, reduction="mean")

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        # Reduce learning rate more gradually
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return [optimizer], [scheduler]

    @torch.no_grad()
    def sample(self, shape, device, temperature=1.0, top_k=None, top_p=0.9):
        """Generate new maps using iterative demasking with better sampling"""
        batch_size, layers, h, w = shape

        # Start with random tokens, but bias toward common tokens
        # Get token frequencies from a reference (you should pass this in)
        x = torch.randint(0, self.vocab_size, (batch_size, layers, h, w), device=device)

        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device)

            # Predict logits
            logits = self(x, t_batch)  # (batch, layers, vocab_size, h, w)

            # Apply temperature
            logits = logits / temperature

            # Get probabilities
            probs = F.softmax(logits, dim=2)

            # Sample new tokens
            probs_flat = probs.permute(0, 1, 3, 4, 2).reshape(-1, self.vocab_size)
            sampled_tokens = torch.multinomial(probs_flat, num_samples=1)
            sampled_tokens = sampled_tokens.reshape(batch_size, layers, h, w)

            if t > 0:
                update_prob = max(
                    0.1, self.mask_probs[t - 1].item() * 0.3
                )  # Much more conservative
                update_mask = (
                    torch.rand(batch_size, layers, h, w, device=device) < update_prob
                )

                x = torch.where(update_mask, sampled_tokens, x)
            else:
                x = sampled_tokens

        return x

    @torch.no_grad()
    def sample_with_guidance(self, shape, device, temperature=1.0, guidance_steps=5):
        """Enhanced sampling with self-guidance"""
        batch_size, layers, h, w = shape

        # Start with random tokens
        x = torch.randint(0, self.vocab_size, (batch_size, layers, h, w), device=device)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device)

            # Multiple guidance steps at each timestep
            for _ in range(guidance_steps if t > self.timesteps // 2 else 1):
                logits = self(x, t_batch)
                logits = logits / temperature
                probs = F.softmax(logits, dim=2)

                # Sample with higher confidence
                probs_flat = probs.permute(0, 1, 3, 4, 2).reshape(-1, self.vocab_size)
                sampled_tokens = torch.multinomial(probs_flat, num_samples=1)
                sampled_tokens = sampled_tokens.reshape(batch_size, layers, h, w)

                if t > 0:
                    # Only update most uncertain positions
                    confidence = torch.max(probs, dim=2)[0]
                    uncertainty_threshold = torch.quantile(confidence.flatten(), 0.3)
                    update_mask = confidence < uncertainty_threshold
                    x = torch.where(update_mask, sampled_tokens, x)
                else:
                    x = sampled_tokens

        return x
