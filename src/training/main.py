import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import subprocess
import datetime

from src.data_processing.datasets import MeshDataset, collate_fn
from src.models.models import BaselineVAE, NemesisVAE
from src.models.decoders import ModulatedResidualNet
from src.models.surrogate_models import BaselineSurrogate, NemesisSurrogate # Import surrogate models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'nemesis'])
    parser.add_argument('--train_mode', type=str, default='vae', choices=['vae', 'surrogate', 'both'], help='Mode of training: vae or surrogate')
    parser.add_argument('--dataset', type=str, default='airfoil', choices=['airfoil'])
    parser.add_argument('--data_dir', type=str, default='data/deepmind-research/meshgraphnets/datasets')
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--num_sdf_points', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d0', type=int, default=256, help='Embedding dimension after PointNet backbone')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=512, help='Feed-forward dimension in transformer blocks')
    parser.add_argument('--n_blocks', type=int, default=4, help='Number of transformer blocks')
    parser.add_argument('--decoder_hidden_dim', type=int, default=256, help='Hidden dimension for decoder MLPs')
    parser.add_argument('--save_model', action='store_true', help='Save the trained models')
    parser.add_argument('--load_vae_model', type=str, default=None, help='Path to load pre-trained VAE models (encoder and decoder)')
    parser.add_argument('--load_surrogate_model', type=str, default=None, help='Path to load pre-trained surrogate model')
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()

    # Setup logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file_path = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    log_file = open(log_file_path, "w")

    def log_message(message):
        print(message)
        log_file.write(message + "\n")

    if args.local:
        args.batch_size = 1
        args.epochs = 1
        args.num_points = 8
        args.num_sdf_points = 8
        # Drastically reduce model dimensions for very fast local runs
        args.d0 = 16 # Reduced embedding dimension
        args.heads = 1 # Single attention head
        args.d_ff = 32 # Reduced feed-forward dimension
        args.n_blocks = 1 # Single transformer block
        args.decoder_hidden_dim = 16 # Drastically reduced decoder hidden dimension

    # Automatically save models if training both VAE and surrogate
    if args.train_mode == 'both':
        args.save_model = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_message(f"Device = {device}")

    # Create the dataset and dataloader
    dataset_path = os.path.join(args.data_dir, args.dataset)
    tfrecord_path = os.path.join(dataset_path, 'train.tfrecord')

    # Check if the dataset exists. If not, instruct the user to run setup_data.sh
    if not os.path.exists(tfrecord_path):
        log_message(f"Error: Dataset not found at {tfrecord_path}.")
        log_message("Please run './setup_data.sh' from the project root to download the dataset.")
        exit(1)

    dataset = MeshDataset(tfrecord_path, args.num_points, args.num_sdf_points)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    encoder = None # Initialize encoder outside the if blocks
    decoder = None # Initialize decoder outside the if blocks
    decoder_latent_dim = None # Initialize decoder_latent_dim outside the if blocks

    # --- VAE Training/Loading Logic ---
    if args.train_mode == 'vae' or args.train_mode == 'both':
        # Create VAE models
        if args.model == 'baseline':
            encoder = BaselineVAE(n_blocks=args.n_blocks, d0=args.d0, heads=args.heads, d_ff=args.d_ff).to(device)
            decoder_latent_dim = args.d0 // 2
        else:
            encoder = NemesisVAE(n_blocks=args.n_blocks, d0=args.d0, heads=args.heads, d_ff=args.d_ff).to(device)
            decoder_latent_dim = (args.d0 // 2) // 2
        decoder = ModulatedResidualNet(in_features=3, out_features=1, latent_dim=decoder_latent_dim, hidden_dim=args.decoder_hidden_dim).to(device)

        log_message(f"\n--- VAE Model Configuration ---")
        log_message(f"Model: {args.model}")
        log_message(f"Encoder Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        log_message(f"Decoder Parameters: {sum(p.numel() for p in decoder.parameters()):,}")
        log_message(f"Total VAE Parameters: {sum(p.numel() for p in list(encoder.parameters()) + list(decoder.parameters())):,}")
        log_message(f"VAE Latent Dimension: {decoder_latent_dim}")
        log_message(f"-------------------------------")

        if args.load_vae_model:
            log_message(f"Loading VAE models from {args.load_vae_model}...")
            checkpoint = torch.load(args.load_vae_model, map_location=device)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            log_message("VAE models loaded successfully. Skipping VAE training.")
        else:
            optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
            log_message("\nStarting VAE training loop...")
            for epoch in range(args.epochs):
                log_message(f"\n--- Epoch {epoch+1}/{args.epochs} (VAE Training) ---")
                for batch_idx, batch in enumerate(dataloader):
                    if args.local and batch_idx > 0:
                        log_message("  (Local mode: Skipping subsequent batches)")
                        break
                    log_message(f"  Processing batch {batch_idx+1}...")
                    points = batch['points'].to(device)
                    normals = batch['normals'].to(device)
                    cells = batch['cells'].to(device)
                    sdf_points = batch['sdf_points'].to(device)
                    sdf_values = batch['sdf_values'].to(device)

                    mu, logvar = encoder(data={'points': points, 'normals': normals, 'cells': cells})
                    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
                    sdf_pred = decoder(sdf_points, z)

                    recon_loss = torch.mean((sdf_pred - sdf_values).pow(2))
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kl_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    log_message(f"  Batch {batch_idx+1} VAE Loss: {loss.item():.4f}")
                log_message(f"Epoch {epoch+1} finished. Final VAE Loss for epoch: {loss.item():.4f}")
            log_message("\nVAE Training loop finished.")

            if args.save_model:
                model_dir = "models"
                os.makedirs(model_dir, exist_ok=True)
                model_name = f"{args.model}_vae_{timestamp}.pth"
                model_path = os.path.join(model_dir, model_name)
                log_message(f"Saving VAE models to {model_path}...")
                torch.save({
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'args': args
                }, model_path)
                log_message("VAE models saved successfully.")

    # --- Surrogate Training/Loading Logic ---
    if args.train_mode == 'surrogate' or args.train_mode == 'both':
        if args.train_mode == 'surrogate':
            if not args.load_vae_model:
                log_message("Error: --load_vae_model must be provided for surrogate training.")
                exit(1)

        # Load pre-trained VAE encoder if not already trained in 'both' mode
        if encoder is None: # This means train_mode was 'surrogate' and VAE was not trained in this run
            log_message(f"Loading pre-trained VAE encoder from {args.load_vae_model} for surrogate training...")
            vae_checkpoint = torch.load(args.load_vae_model, map_location=device)
            vae_args = vae_checkpoint['args']

            if vae_args.model == 'baseline': # This refers to the VAE model type that produced the latent space
                encoder = BaselineVAE(n_blocks=vae_args.n_blocks, d0=vae_args.d0, heads=vae_args.heads, d_ff=vae_args.d_ff).to(device)
                surrogate_latent_dim = vae_args.d0 // 2
            else:
                encoder = NemesisVAE(n_blocks=vae_args.n_blocks, d0=vae_args.d0, heads=vae_args.heads, d_ff=vae_args.d_ff).to(device)
                surrogate_latent_dim = (vae_args.d0 // 2) // 2
            encoder.load_state_dict(vae_checkpoint['encoder_state_dict'])
        else: # Encoder was just trained in 'both' mode
            log_message("Using newly trained VAE encoder for surrogate training.")
            if args.model == 'baseline':
                surrogate_latent_dim = args.d0 // 2
            else:
                surrogate_latent_dim = (args.d0 // 2) // 2

        encoder.eval() # Set encoder to evaluation mode
        for param in encoder.parameters(): # Freeze encoder parameters
            param.requires_grad = False

        # Create Surrogate model
        if args.model == 'baseline': # This refers to the VAE model type that produced the latent space
            surrogate_model = BaselineSurrogate(latent_dim=surrogate_latent_dim, n_blocks=args.n_blocks, heads=args.heads, d_ff=args.d_ff).to(device)
        else:
            surrogate_model = NemesisSurrogate(latent_dim=surrogate_latent_dim, n_blocks=args.n_blocks, heads=args.heads, d_ff=args.d_ff).to(device)
        
        log_message(f"\n--- Surrogate Model Configuration ---")
        log_message(f"Surrogate Model Type: {args.model}")
        log_message(f"Surrogate Parameters: {sum(p.numel() for p in surrogate_model.parameters()):,}")
        log_message(f"Input Latent Dimension: {surrogate_latent_dim}")
        log_message(f"-------------------------------------")

        if args.load_surrogate_model:
            log_message(f"Loading surrogate model from {args.load_surrogate_model}...")
            surrogate_model.load_state_dict(torch.load(args.load_surrogate_model, map_location=device))
            log_message("Surrogate model loaded successfully. Skipping surrogate training.")
        else:
            optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=args.lr)
            criterion = nn.L1Loss() # MAE Loss

            log_message("\nStarting Surrogate training loop...")
            for epoch in range(args.epochs):
                log_message(f"\n--- Epoch {epoch+1}/{args.epochs} (Surrogate Training) ---")
                for batch_idx, batch in enumerate(dataloader):
                    if args.local and batch_idx > 0:
                        log_message("  (Local mode: Skipping subsequent batches)")
                        break
                    log_message(f"  Processing batch {batch_idx+1}...")
                    points = batch['points'].to(device)
                    normals = batch['normals'].to(device)
                    cells = batch['cells'].to(device)
                    labels = batch['label'].to(device) # Target physical property

                    with torch.no_grad(): # Ensure VAE encoder is not trained
                        mu, logvar = encoder(data={'points': points, 'normals': normals, 'cells': cells})
                        z = mu # Use mu as the latent vector for surrogate training

                    predictions = surrogate_model(z)
                    loss = criterion(predictions, labels.unsqueeze(1))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    log_message(f"  Batch {batch_idx+1} Surrogate MAE Loss: {loss.item():.4f}")
                log_message(f"Epoch {epoch+1} finished. Final Surrogate MAE Loss for epoch: {loss.item():.4f}")
            log_message("\nSurrogate Training loop finished.")

            if args.save_model:
                model_dir = "models"
                os.makedirs(model_dir, exist_ok=True)
                model_name = f"{args.model}_surrogate_{timestamp}.pth"
                model_path = os.path.join(model_dir, model_name)
                log_message(f"Saving surrogate model to {model_path}...")
                torch.save(surrogate_model.state_dict(), model_path)
                log_message("Surrogate model saved successfully.")

    log_file.close()

if __name__ == '__main__':
    main()
