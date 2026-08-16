import torch
import torch.nn as nn
from transformers import RobertaModel, GPT2LMHeadModel

class HAMVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(HAMVAE, self).__init__()
        # Encoder: RoBERTa
        self.encoder = RobertaModel.from_pretrained("roberta-base")
        self.fc_mu = nn.Linear(self.encoder.config.hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder.config.hidden_size, latent_dim)

        # Decoder: GPT-2
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + eps*std"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_ids, attention_mask, labels=None):
        # Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token embedding

        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        z = self.reparameterize(mu, logvar)

        # Decode (GPT-2 language modeling)
        decoder_outputs = self.decoder(input_ids, labels=labels)

        return decoder_outputs.loss, decoder_outputs.logits
