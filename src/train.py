import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class GPSTripletLoss(nn.Module):
    """
    Computes Triplet Loss based on physical GPS distance.
    """
    def __init__(self, margin=0.3, pos_thresh=15.0, neg_thresh=50.0):
        super().__init__()
        self.margin = margin
        self.pos_thresh = pos_thresh 
        self.neg_thresh = neg_thresh

    def forward(self, embeddings, gps_coords):
        """
        embeddings: [Batch, 512]
        gps_coords: [Batch, 2] (Assumed projected x,y or similar scale)
        """
        # 1. Compute pairwise distance matrices
        dist_emb = torch.cdist(embeddings, embeddings, p=2)
        dist_gps = torch.cdist(gps_coords, gps_coords, p=2)
        
        # 2. Create Masks
        # Positive: dist_gps < 15m AND not self (diagonal)
        mask_pos = (dist_gps < self.pos_thresh) & (dist_gps > 0)
        # Negative: dist_gps > 50m
        mask_neg = (dist_gps > self.neg_thresh)
        
        triplet_loss = torch.tensor(0.0, device=embeddings.device)
        num_valid_triplets = 0
        
        # 3. Mining Loop 
        for i in range(embeddings.shape[0]):
            pos_indices = mask_pos[i].nonzero(as_tuple=True)[0]
            neg_indices = mask_neg[i].nonzero(as_tuple=True)[0]
            
            if len(pos_indices) > 0 and len(neg_indices) > 0:
                # We want to pull the farthest positive closer
                current_emb_dists = dist_emb[i]
                hardest_pos_idx = pos_indices[torch.argmax(current_emb_dists[pos_indices])]
                hardest_pos_dist = current_emb_dists[hardest_pos_idx]
                
                # We want to push the nearest negative away
                hardest_neg_idx = neg_indices[torch.argmin(current_emb_dists[neg_indices])]
                hardest_neg_dist = current_emb_dists[hardest_neg_idx]
                
                # Calculate Loss
                loss = torch.relu(hardest_pos_dist**2 - hardest_neg_dist**2 + self.margin)
                
                if loss > 0:
                    triplet_loss += loss
                    num_valid_triplets += 1
                    
        if num_valid_triplets > 0:
            return triplet_loss / num_valid_triplets
        else:
            return triplet_loss # Returns 0 if no valid triplets found in batch

def train_model(model, dataloader, epochs=100, device='cuda'):
    model = model.to(device)
    
    # Optimizer specs from your text
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_triplet = GPSTripletLoss(margin=0.3, pos_thresh=15, neg_thresh=50)
    
    lambda_weight = 1.0

    print("Starting training...")
    pbar = tqdm(epoch)
    for epoch in pbar:
        model.train()
        total_loss = 0
        pbar.set_description(f"Loss:{loss_cls / len(dataloader):.4f}, {loss_trip / len(dataloader):.4f} ")

        for batch_idx, (images, labels, gps) in enumerate(dataloader):
            images, labels, gps = images.to(device), labels.to(device), gps.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            cls_out, emb_out = model(images)
            
            # Compute Losses
            loss_cls = criterion_cls(cls_out, labels)
            loss_trip = criterion_triplet(emb_out, gps)
            
            # Combine
            loss_total = loss_cls + (lambda_weight * loss_trip)
            
            # Backward
            loss_total.backward()
            optimizer.step()
            
            total_loss += loss_total.item()
            
        scheduler.step()
    return model