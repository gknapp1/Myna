import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch.nn.functional as F


def _augment_batch_random(batch_inputs: torch.Tensor) -> torch.Tensor:
    # Circular Depth Shift (a->b->c is like b->c->a)
    depth = batch_inputs.shape[4]
    shift = torch.randint(0, depth, (1,)).item()
    augmented_batch = torch.roll(batch_inputs, shifts=shift, dims=2)

    # Random 90-degree Rotations in the H-W plane
    k = torch.randint(0, 4, (1,)).item()  # 0, 1, 2, or 3 rotations
    augmented_batch = torch.rot90(augmented_batch, k, dims=[2, 3])

    # Random Flip
    if torch.rand(1) > 0.5:
        augmented_batch = torch.flip(augmented_batch, dims=[3])

    return augmented_batch


def _augment_batch_exhaustive(
    batch_inputs: torch.Tensor, batch_targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

    # Get shapes
    D = batch_inputs.shape[4]
    num_augmentations = D * 4 * 2  #  D depth shifts, 4 rotations, 2 flip states

    # Use a list to collect all transformed versions of the batch
    all_augmented_inputs = []

    # Loop through all possible depth shifts
    for shift in range(D):
        shifted_batch = torch.roll(batch_inputs, shifts=shift, dims=4)
        for k in range(4):  # 0, 1, 2, or 3 rotations
            rotated_batch = torch.rot90(shifted_batch, k, dims=[2, 3])
            # Add the unflipped and flipped version
            all_augmented_inputs.append(rotated_batch)
            flipped_batch = torch.flip(rotated_batch, dims=[3])
            all_augmented_inputs.append(flipped_batch)

    # Concatenate all augmented batches
    augmented_inputs = torch.cat(all_augmented_inputs, dim=0)

    # The target is the same for all augmentations of a given sample.
    augmented_targets = batch_targets.repeat(num_augmentations, 1)

    return augmented_inputs, augmented_targets


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    beta: float,
    device: torch.device,
):

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Containers for losses
    train_loss_list, test_loss_list = [], []

    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        # Train
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            # Permute to PyTorch's expected format: (N, C, D, H, W).
            # batch_inputs = batch_inputs.permute(0, 1, 4, 2, 3)

            # Augment batch with all possible permutations
            augmented_inputs, augmented_targets = _augment_batch_exhaustive(
                batch_inputs, batch_targets
            )

            # Forward pass
            mu, log_var, z_sample = model(augmented_inputs)

            # Calculate Gaussian NLL Loss
            variance = torch.exp(log_var) + 1e-8  # Add epsilon for stability
            log_term = 0.5 * torch.log(variance)
            error_term = (
                0.5 * F.mse_loss(mu, augmented_targets, reduction="none") / variance
            )
            loss = torch.mean(log_term + error_term)

            # # Calculate VAE loss
            # reconstruction_loss = F.mse_loss(z_sample, augmented_targets)
            # # Average KL divergence over the batch
            # kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            # loss = reconstruction_loss + beta * kl_divergence

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)

                mu, log_var, z_sample = model(batch_inputs)

                # Calculate Gaussian NLL Loss
                variance = torch.exp(log_var) + 1e-8  # Add epsilon for stability
                log_term = 0.5 * torch.log(variance)
                error_term = (
                    0.5 * F.mse_loss(mu, batch_targets, reduction="none") / variance
                )
                loss = torch.mean(log_term + error_term)

                # # Calculate VAE loss
                # reconstruction_loss = F.mse_loss(mu, batch_targets)
                # kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                # loss = reconstruction_loss + beta * kl_divergence

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Append to loss containers
        train_loss_list.append(avg_train_loss)
        test_loss_list.append(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs} -> Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}"
        )

    print("--- Training Complete ---")

    return train_loss_list, test_loss_list
