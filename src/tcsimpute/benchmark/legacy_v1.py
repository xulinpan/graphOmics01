# Verbatim copy of the original (v1) model, kept only to reproduce the "v1 model" comparator.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer to incorporate biological priors 
    (e.g., Protein-Protein Interaction networks or Gene Regulatory Networks).
    """
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, embed_dim)
        
    def forward(self, x, adj):
        # x: [batch_size, num_features]
        # adj: [batch_size, num_features, num_features] (Batched by DataLoader)
        
        # Safely handle batch matrix multiplication to prevent 4D broadcasting
        if adj.dim() == 3:
            # [B, 1, F] * [B, F, F] -> [B, 1, F] -> [B, F]
            support = torch.bmm(x.unsqueeze(1), adj).squeeze(1)
        else:
            support = torch.matmul(x, adj)
            
        out = self.linear(support)
        return F.relu(out)

class GraphOmicsEmbedding(nn.Module):
    """
    Projects different omics layers into a shared space using GCNs.
    """
    def __init__(self, input_dims, embed_dim):
        super().__init__()
        self.gcn_layers = nn.ModuleList([
            GCNLayer(dim, embed_dim) for dim in input_dims
        ])
        
    def forward(self, x_list, adj_list):
        embeddings = []
        for gcn, x, adj in zip(self.gcn_layers, x_list, adj_list):
            emb = gcn(x, adj).unsqueeze(1) # [batch, 1, embed_dim]
            embeddings.append(emb)
        return torch.cat(embeddings, dim=1) # [batch, num_omics, embed_dim]

class MultiOmicsTransformerVAE(nn.Module):
    def __init__(self, input_dims, embed_dim=256, num_heads=8, num_layers=3, latent_dim=128):
        super().__init__()
        self.num_omics = len(input_dims)
        self.input_dims = input_dims
        
        # 1. Graph-Guided Omics Embedding
        self.embedding = GraphOmicsEmbedding(input_dims, embed_dim)
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. VAE Latent Space
        self.fc_mu = nn.Linear(self.num_omics * embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.num_omics * embed_dim, latent_dim)
        
        # 4. Decoder
        self.decoder_input = nn.Linear(latent_dim, self.num_omics * embed_dim)
        self.reconstructors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, dim)
            ) for dim in input_dims
        ])

    def encode(self, x_list, adj_list, mask_list=None):
        if mask_list is not None:
            x_list = [x * mask for x, mask in zip(x_list, mask_list)]
            
        x_emb = self.embedding(x_list, adj_list)
        encoded = self.transformer_encoder(x_emb)
        
        # Save contextualized embeddings for contrastive loss
        self.contextualized_embs = encoded 
        
        encoded_flat = encoded.view(encoded.size(0), -1)
        mu = self.fc_mu(encoded_flat)
        logvar = self.fc_logvar(encoded_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        dec_hidden = self.decoder_input(z)
        dec_hidden = dec_hidden.view(dec_hidden.size(0), self.num_omics, -1)
        
        reconstructed = []
        for i in range(self.num_omics):
            omics_slice = dec_hidden[:, i, :]
            reconstructed.append(self.reconstructors[i](omics_slice))
        return reconstructed

    def forward(self, x_list, adj_list, mask_list=None):
        mu, logvar = self.encode(x_list, adj_list, mask_list)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar, self.contextualized_embs

def contrastive_loss(embs, temperature=0.1):
    """
    InfoNCE Contrastive Loss to align modalities (e.g., RNA and Protein) for the same patient.
    embs: [batch_size, num_omics, embed_dim]
    """
    # Aligning modality 0 (RNA) and modality 1 (Protein)
    h_rna = F.normalize(embs[:, 0, :], dim=1)
    h_prot = F.normalize(embs[:, 1, :], dim=1)
    
    similarity = torch.matmul(h_rna, h_prot.T) / temperature
    labels = torch.arange(h_rna.size(0)).to(h_rna.device)
    return F.cross_entropy(similarity, labels)

def left_censored_loss(recon, orig, mask, tau=-1.0):
    """
    Handles Not Missing At Random (NMAR) data.
    Penalizes the model if it predicts high abundance for missing values,
    assuming missingness is due to the value being below a detection threshold (tau).
    """
    missing_mask = 1.0 - mask
    # Penalty: max(0, predicted - threshold)^2 for missing values
    penalty = F.relu(recon - tau) ** 2
    censored_loss = (penalty * missing_mask).sum() / missing_mask.sum().clamp(min=1.0)
    return censored_loss

def state_of_the_art_loss(reconstructed, original, masks, mu, logvar, contextualized_embs, beta=1.0, gamma=0.5, lambda_c=0.1, censor_thresholds=None):
    """
    Comprehensive SOTA Loss Function combining:
    1. MSE on observed data
    2. Left-censored loss on missing data (NMAR)
    3. KL Divergence (Bayesian regularization)
    4. Contrastive Alignment Loss
    """
    recon_loss = 0
    censor_loss = 0
    
    if censor_thresholds is None:
        censor_thresholds = [-0.5] * len(reconstructed)
    if len(censor_thresholds) != len(reconstructed):
        raise ValueError("Provide one censoring threshold (scalar or feature vector) per modality.")

    for recon, orig, mask, threshold in zip(reconstructed, original, masks, censor_thresholds):
        # 1. Standard MSE on observed data
        mse = F.mse_loss(recon * mask, orig * mask, reduction='sum')
        recon_loss += mse / mask.sum().clamp(min=1.0)
        
        # 2. Left-censored loss on missing data
        censor_loss += left_censored_loss(recon, orig, mask, tau=threshold)
        
    # 3. KL Divergence
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 4. Contrastive Alignment (RNA vs Protein)
    cont_loss = contrastive_loss(contextualized_embs)
    
    total_loss = recon_loss + (gamma * censor_loss) + (beta * kld_loss) + (lambda_c * cont_loss)
    return total_loss, recon_loss, censor_loss, kld_loss, cont_loss
