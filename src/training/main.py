import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import subprocess
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from src.data_processing.datasets import MeshDataset, collate_fn
from src.models.models import BaselineVAE, NemesisVAE
from src.models.decoders import ModulatedResidualNet
from src.models.surrogate_models import BaselineSurrogate, NemesisSurrogate # Import surrogate models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'nemesis'])
    parser.add_argument(
        '--train_mode',
        type=str,
        default='vae',
        choices=['vae', 'surrogate', 'both', 'flow_inverse'],
        help='Mode of training: vae, surrogate, joint, or inverse flow',
    )
    parser.add_argument('--dataset', type=str, default='airfoil', choices=['airfoil'])
    parser.add_argument('--data_dir', type=str, default='data/deepmind-research/meshgraphnets/datasets')
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--num_sdf_points', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for VAE training')
    parser.add_argument('--surrogate_epochs', type=int, default=10, help='Number of epochs for Surrogate training')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d0', type=int, default=128, help='Embedding dimension after PointNet backbone')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=256, help='Feed-forward dimension in transformer blocks')
    parser.add_argument('--n_blocks', type=int, default=4, help='Number of transformer blocks')
    parser.add_argument('--scale', type=str, default='1x', choices=['1x', '3x'], help='Model parameter scale (1x or 3x)')
    parser.add_argument('--decoder_hidden_dim', type=int, default=128, help='Hidden dimension for decoder MLPs')
    parser.add_argument('--save_model', action='store_true', help='Save the trained models')
    parser.add_argument('--load_vae_model', type=str, default=None, help='Path to load pre-trained VAE models (encoder and decoder)')
    parser.add_argument('--load_surrogate_model', type=str, default=None, help='Path to load pre-trained surrogate model')
    parser.add_argument('--flow_dataset', type=str, default=None, help='Path or identifier for inverse flow training dataset')
    parser.add_argument('--flow_batch_size', type=int, default=8, help='Batch size for flow inverse training')
    parser.add_argument('--flow_epochs', type=int, default=20, help='Epochs for flow inverse training')
    parser.add_argument('--flow_lr', type=float, default=1e-4, help='Learning rate for flow inverse trainer')
    parser.add_argument('--flow_latent_dim', type=int, default=128, help='Latent dimension for flow inverse encoder')
    parser.add_argument('--flow_hidden_dim', type=int, default=256, help='Hidden dimension for flow inverse models')
    parser.add_argument('--flow_condition_dim', type=int, default=None, help='Conditioning dimension for flow inverse networks')
    parser.add_argument('--flow_loss_geometry_weight', type=float, default=1.0, help='Weight for geometry reconstruction loss')
    parser.add_argument('--flow_loss_flow_weight', type=float, default=0.1, help='Weight for flow consistency loss')
    parser.add_argument('--flow_loss_latent_weight', type=float, default=1e-4, help='Weight for latent regularization term')
    parser.add_argument('--flow_checkpoint_dir', type=str, default=None, help='Directory to store flow inverse checkpoints')
    parser.add_argument('--flow_resume_path', type=str, default=None, help='Checkpoint path to resume flow inverse training')
    parser.add_argument('--flow_log_every', type=int, default=10, help='Logging frequency (in batches) for flow inverse training')
    parser.add_argument('--flow_val_split', type=float, default=0.1, help='Fraction of data reserved for validation in flow inverse training')
    parser.add_argument('--flow_scheduler_step', type=int, default=None, help='Step interval for flow inverse learning rate scheduler')
    parser.add_argument('--flow_scheduler_gamma', type=float, default=0.5, help='Gamma for flow inverse learning rate scheduler')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--reynolds_number', type=float, default=None, help='Optional Reynolds number conditioning for flow inverse training')
    args = parser.parse_args()

    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = os.path.join("logs", f"run_{timestamp}")
    os.makedirs(run_log_dir, exist_ok=True)
    log_file_path = os.path.join(run_log_dir, f"training_log_{timestamp}.txt")
    log_file = open(log_file_path, "w")

    def log_message(message):
        print(message)
        log_file.write(message + "\n")

    if args.local:
        args.batch_size = 1
        args.epochs = 1 # VAE epochs for local mode
        args.surrogate_epochs = 1 # Surrogate epochs for local mode
        args.num_points = 8
        args.num_sdf_points = 8
        # Drastically reduce model dimensions for very fast local runs
        args.d0 = 16 # Reduced embedding dimension
        args.heads = 1 # Single attention head
        args.d_ff = 32 # Reduced feed-forward dimension
        args.n_blocks = 1 # Single transformer block
        args.decoder_hidden_dim = 16 # Drastically reduced decoder hidden dimension
        args.flow_batch_size = 1
        args.flow_epochs = 1
        args.flow_latent_dim = min(args.flow_latent_dim, 16)
        args.flow_hidden_dim = min(args.flow_hidden_dim, 32)
    else:
        # Apply scaling based on args.scale
        if args.scale == '3x':
            args.d0 = args.d0 * 2
            args.heads = args.heads * 2 # Can be adjusted based on desired scaling
            args.d_ff = args.d_ff * 2
            args.n_blocks = args.n_blocks * 2
            args.decoder_hidden_dim = args.decoder_hidden_dim * 2

    # Automatically save models if training both VAE and surrogate
    if args.train_mode == 'both':
        args.save_model = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_message(f"Device = {device}")

    if args.train_mode == 'flow_inverse':
        from src.training.inverse_trainer import run_flow_inverse_training

        run_flow_inverse_training(args, log_message, device)
        log_file.close()
        return

    # Ensure dataset is downloaded
    dataset_path = os.path.join(args.data_dir, args.dataset)
    tfrecord_path = os.path.join(dataset_path, 'train.tfrecord')

    # Check if the dataset exists. If not, instruct the user to run setup_data.sh
    if not os.path.exists(tfrecord_path):
        log_message(f"Error: Dataset not found at {tfrecord_path}.")
        log_message("Please run './setup_data.sh' from the project root to download the dataset.")
        exit(1)

    # Create the full dataset and split into train/validation
    full_dataset = MeshDataset(tfrecord_path, args.num_points, args.num_sdf_points, is_local=args.local, batch_size=args.batch_size)
    
    # Determine split sizes
    total_samples = len(full_dataset)
    if args.local:
        # For local mode, ensure validation set is at least 1 sample if possible
        train_size = max(1, int(total_samples * 0.8))
        val_size = total_samples - train_size
        if val_size == 0 and total_samples > 0: # Ensure at least one sample for validation if possible
            train_size = total_samples - 1
            val_size = 1
        elif total_samples == 0:
            train_size = 0
            val_size = 0
    else:
        train_size = int(0.8 * total_samples)
        val_size = total_samples - train_size

    if total_samples == 0:
        log_message("Error: Dataset is empty. Cannot proceed with training.")
        exit(1)

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    log_message(f"\nNumber of training batches per epoch: {len(train_dataloader)}")
    log_message(f"Number of validation batches per epoch: {len(val_dataloader)}")

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
            
            vae_train_losses = []
            vae_val_losses = []

            log_message("\nStarting VAE training loop...")
            for epoch in range(args.epochs):
                log_message(f"\n--- Epoch {epoch+1}/{args.epochs} (VAE Training) ---")
                # Training
                encoder.train()
                decoder.train()
                total_train_loss = 0
                for batch_idx, batch in enumerate(train_dataloader):
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

                    recon_loss = torch.mean((sdf_pred - sdf_values.unsqueeze(-1)).pow(2))
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kl_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.item()
                    log_message(f"  Batch {batch_idx+1} VAE Loss: {loss.item():.4f}")
                avg_train_loss = total_train_loss / (batch_idx + 1)
                vae_train_losses.append(avg_train_loss)

                # Validation
                encoder.eval()
                decoder.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch_idx_val, batch_val in enumerate(val_dataloader):
                        if args.local and batch_idx_val > 0:
                            break
                        points_val = batch_val['points'].to(device)
                        normals_val = batch_val['normals'].to(device)
                        cells_val = batch_val['cells'].to(device)
                        sdf_points_val = batch_val['sdf_points'].to(device)
                        sdf_values_val = batch_val['sdf_values'].to(device)

                        mu_val, logvar_val = encoder(data={'points': points_val, 'normals': normals_val, 'cells': cells_val})
                        z_val = mu_val + torch.exp(0.5 * logvar_val) * torch.randn_like(logvar_val)
                        sdf_pred_val = decoder(sdf_points_val, z_val)

                        recon_loss_val = torch.mean((sdf_pred_val - sdf_values_val.unsqueeze(-1)).pow(2))
                        kl_loss_val = -0.5 * torch.mean(1 + logvar_val - mu_val.pow(2) - logvar_val.exp())
                        loss_val = recon_loss_val + kl_loss_val
                        total_val_loss += loss_val.item()
                avg_val_loss = total_val_loss / (batch_idx_val + 1)
                vae_val_losses.append(avg_val_loss)

                log_message(f"Epoch {epoch+1} finished. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            log_message("\nVAE Training loop finished.")

            # Plot VAE Convergence
            plt.figure(figsize=(10, 5))
            plt.plot(vae_train_losses, label='Train Loss')
            plt.plot(vae_val_losses, label='Validation Loss')
            plt.title('VAE Convergence')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(run_log_dir, 'vae_convergence.png'))
            plt.close()
            log_message(f"VAE convergence plot saved to {os.path.join(run_log_dir, 'vae_convergence.png')}")

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

            surrogate_train_losses = []
            surrogate_val_losses = []
            surrogate_val_r2_scores = []
            surrogate_val_mape_scores = []

            log_message("\nStarting Surrogate training loop...")
            for epoch in range(args.surrogate_epochs):
                log_message(f"\n--- Epoch {epoch+1}/{args.surrogate_epochs} (Surrogate Training) ---")
                # Training
                surrogate_model.train()
                total_train_loss = 0
                for batch_idx, batch in enumerate(train_dataloader):
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

                    total_train_loss += loss.item()
                    log_message(f"  Batch {batch_idx+1} Surrogate MAE Loss: {loss.item():.4f}")
                avg_train_loss = total_train_loss / (batch_idx + 1)
                surrogate_train_losses.append(avg_train_loss)

                # Validation
                surrogate_model.eval()
                total_val_loss = 0
                all_predictions = []
                all_labels = []
                with torch.no_grad():
                    for batch_idx_val, batch_val in enumerate(val_dataloader):
                        if args.local and batch_idx_val > 0:
                            break
                        points_val = batch_val['points'].to(device)
                        normals_val = batch_val['normals'].to(device)
                        cells_val = batch_val['cells'].to(device)
                        labels_val = batch_val['label'].to(device)

                        mu_val, logvar_val = encoder(data={'points': points_val, 'normals': normals_val, 'cells': cells_val})
                        z_val = mu_val

                        predictions_val = surrogate_model(z_val)
                        loss_val = criterion(predictions_val, labels_val.unsqueeze(1))
                        total_val_loss += loss_val.item()

                        all_predictions.extend(predictions_val.cpu().numpy().flatten())
                        all_labels.extend(labels_val.cpu().numpy().flatten())
                
                # Calculate Mean Absolute Percentage Error (MAPE)
                # Add a small epsilon to avoid division by zero
                epsilon = 1e-8 
                mape = np.mean(np.abs((np.array(all_labels) - np.array(all_predictions)) / (np.array(all_labels) + epsilon))) * 100
                surrogate_val_mape_scores.append(mape)
                
                avg_val_loss = total_val_loss / (batch_idx_val + 1)
                surrogate_val_losses.append(avg_val_loss)
                
                # Calculate R-squared
                r2 = r2_score(all_labels, all_predictions)
                surrogate_val_r2_scores.append(r2)

                log_message(f"Epoch {epoch+1} finished. Train MAE: {avg_train_loss:.4f}, Val MAE: {avg_val_loss:.4f}, Val R^2: {r2:.4f}, Val MAPE: {mape:.2f}%")
            log_message("\nSurrogate Training loop finished.")

            # Plot Surrogate Convergence
            plt.figure(figsize=(10, 5))
            plt.plot(surrogate_train_losses, label='Train MAE')
            plt.plot(surrogate_val_losses, label='Validation MAE')
            plt.title('Surrogate Model Convergence (MAE)')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(run_log_dir, 'surrogate_mae_convergence.png'))
            plt.close()
            log_message(f"Surrogate MAE convergence plot saved to {os.path.join(run_log_dir, 'surrogate_mae_convergence.png')}")

            plt.figure(figsize=(10, 5))
            plt.plot(surrogate_val_r2_scores, label='Validation R^2', color='green')
            plt.title('Surrogate Model R^2 Score')
            plt.xlabel('Epoch')
            plt.ylabel('R^2 Score')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(run_log_dir, 'surrogate_r2_score.png'))
            plt.close()
            log_message(f"Surrogate R^2 score plot saved to {os.path.join(run_log_dir, 'surrogate_r2_score.png')}")





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