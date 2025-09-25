# Base imports
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

from myna.core.app import MynaApp
from .construct_training_dataset import make_dataset
from .neural_net.architecture import Periodic3DCNN
from .neural_net.trainer import train_model
from .neural_net.normalizer import DatasetNormalizer


class GrainStatPredictorTrainerApp(MynaApp):
    """Application for training a neural network to predict grain statistics"""

    def __init__(self, name="grain_stat_predictor_trainer"):
        super().__init__(name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_file = "dataset.pt"
        self.data_file_training = "data_training.pt"
        self.data_file_validation = "data_validation.pt"
        self.data_file_normalizer = "normalizer.pt"
        self.model_arg_dict = "model_arg_dict.pt"
        self.trained_model_state_dict = "trained_model_state_dict.pt"
        self.trained_model_loss_dict = "trained_model_loss_dict.pt"

    def format_output_dict(self):
        """Format the output dictionary for the Myna step"""
        outdict = {
            "device": str(self.device),
            "dataset_file": self.dataset_file,
            "data_file_training": self.data_file_training,
            "data_file_validation": self.data_file_validation,
            "data_file_normalizer": self.data_file_normalizer,
            "model_arg_dict": self.model_arg_dict,
            "trained_model_state_dict": self.trained_model_state_dict,
            "trained_model_loss_dict": self.trained_model_loss_dict,
        }
        return outdict

    def parse_configure_arguments(self):
        """Check for arguments relevant to the configure step and update app settings"""
        # Parse app-specific arguments
        self.parser.add_argument(
            "--training-data-dir",
            default=None,
            type=str,
            help="Path to directory containing training data",
        )
        self.parser.add_argument(
            "--test-train-split",
            default=0.8,
            type=float,
            help="Fractional split of dataset for testing and training,"
            " for example, 0.8 = 80/20 split",
        )
        self.args, _ = self.parser.parse_known_args()
        self.mpiargs_to_current()

        # Update derived parameters
        self.set_procs()
        self.set_template_path("pytorch", "grain_stat_predictor_trainer")

    def parse_execute_arguments(self):
        """Check for arguments relevant to the configure step and update app settings"""
        # Parse app-specific arguments
        self.parser.add_argument(
            "--batch-size",
            default=16,
            type=int,
            help="Batch size for neural network training",
        )
        self.parser.add_argument(
            "--epochs",
            default=50,
            type=int,
            help="Number of epochs for neural network training",
        )
        self.parser.add_argument(
            "--learning-rate",
            default=1.0e-4,
            type=float,
            help="Learning rate for neural network training",
        )
        self.parser.add_argument(
            "--beta",
            default=0.1,
            type=float,
            help="Weight for the KL divergence during neural network training",
        )
        self.parser.add_argument(
            "--no-training",
            dest="train_model",
            default=True,
            action="store_false",
            help="Flag to use a pre-trained model",
        )
        self.args, _ = self.parser.parse_known_args()
        self.mpiargs_to_current()

        # Update derived parameters
        self.set_procs()
        self.set_template_path("pytorch", "grain_stat_predictor")

    def configure(self):
        """Configure the training data for the model"""
        # Get command line arguments
        self.parse_configure_arguments()

        # Make or get pytorch dataset
        # "~/mnt/savitar/ConceptLaserM2-ORNL1/2024/01/2024-01-26 M2_AMMT_DOE_10/Myna/cases_ammt_doe_10"
        base_directory = Path(self.args.training_data_dir.strip("'\"")).expanduser()
        inputs_tensor, outputs_tensor = make_dataset(
            base_directory, dataset_path=Path(self.dataset_file).expanduser()
        )

        # Normalize data
        normalizer = DatasetNormalizer(
            input_linear_channels=[2, 3, 6],
            input_log_channels=[0, 1, 4, 5],
            output_linear_channels=[],
            output_log_channels=[0, 1, 2],
            quantile_clip=0.01,
        )
        normalizer.fit(train_inputs=inputs_tensor, train_outputs=outputs_tensor)
        norm_inputs_tensor, norm_outputs_tensor = normalizer.transform(
            inputs=inputs_tensor, outputs=outputs_tensor
        )
        norm_inputs_tensor = torch.nan_to_num(norm_inputs_tensor, nan=0.0)
        norm_outputs_tensor = torch.nan_to_num(norm_outputs_tensor, nan=0.0)

        # Save the normalizer to a file
        torch.save(normalizer, self.data_file_normalizer)

        # Create the full dataset and then split dataset into training
        # and validation sets
        full_dataset = TensorDataset(
            norm_inputs_tensor.to(self.device), norm_outputs_tensor.to(self.device)
        )
        train_split = self.args.test_train_split
        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Save training and validation sets
        torch.save(train_dataset, self.data_file_training)
        torch.save(val_dataset, self.data_file_validation)

    def execute(self):
        "Train the model"

        # Get command line arguments
        self.parse_execute_arguments()

        # Allow checkpoint load
        with torch.serialization.safe_globals(
            [
                torch.utils.data.dataset.Subset,
                torch.utils.data.dataset.TensorDataset,
                DatasetNormalizer,
            ]
        ):

            # Load the training and validation datasets and the normalizer
            train_dataset = torch.load(self.data_file_training)
            val_dataset = torch.load(self.data_file_validation)
            normalizer = torch.load(self.data_file_normalizer)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.args.batch_size, shuffle=False
        )

        # Make the neural network
        model_args = {
            "in_channels": 7,
            "out_features": 3,
            "activation_fn": torch.nn.SiLU,
        }
        torch.save(model_args, self.model_arg_dict)
        model_silu = Periodic3DCNN(**model_args).to(self.device)

        # If using pretrained model, load the state dict, otherwise train the model
        if self.args.train_model:
            # Train the model
            train_loss, test_loss = train_model(
                model=model_silu,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=self.args.epochs,
                learning_rate=self.args.learning_rate,
                beta=self.args.beta,
                device=self.device,
            )
            # Save the model state
            torch.save(model_silu.state_dict(), self.trained_model_state_dict)
            torch.save(
                {"train_loss": train_loss, "test_loss": test_loss},
                self.trained_model_loss_dict,
            )
        else:
            # Load the pre-trained model state
            model_silu.load_state_dict(torch.load(self.trained_model_state_dict))
            # Load the training and testing loss information
            loss_dict = torch.load(self.trained_model_loss_dict)
            train_loss = loss_dict["train_loss"]
            test_loss = loss_dict["test_loss"]

        # Validate the model
        model_silu.eval()
        with torch.no_grad():
            # Get predictions
            targets_real_all = np.array([])
            preds_real_all = np.array([])
            preds_low_all = np.array([])
            preds_high_all = np.array([])
            for batch_inputs, batch_targets in val_loader:
                # Run the model
                mu, log_var, _ = model_silu(batch_inputs)

                # Convert both prediction and target back to their original "real" scale
                mu_cpu, std_cpu = mu.cpu(), torch.exp(0.5 * log_var.cpu())

                # Make low, actual, and high predictions (+/- one std)
                preds_low = normalizer.inverse_transform_outputs(
                    mu_cpu - std_cpu
                ).numpy()
                preds_real = normalizer.inverse_transform_outputs(mu_cpu).numpy()
                # preds_real = normalizer.inverse_transform_outputs(z_sample.cpu()).numpy()
                preds_high = normalizer.inverse_transform_outputs(
                    mu_cpu + std_cpu
                ).numpy()

                # Get targets
                targets_norm = batch_targets.cpu()
                targets_real = normalizer.inverse_transform_outputs(
                    targets_norm
                ).numpy()

                # Append for all batches
                targets_real_all = np.vstack(
                    (
                        targets_real_all.reshape(
                            targets_real_all.shape[0], targets_real.shape[1]
                        ),
                        targets_real,
                    )
                )
                preds_real_all = np.vstack(
                    (
                        preds_real_all.reshape(
                            preds_real_all.shape[0], preds_real.shape[1]
                        ),
                        preds_real,
                    )
                )
                preds_low_all = np.vstack(
                    (
                        preds_low_all.reshape(
                            preds_low_all.shape[0], preds_low.shape[1]
                        ),
                        preds_low,
                    )
                )
                preds_high_all = np.vstack(
                    (
                        preds_high_all.reshape(
                            preds_high_all.shape[0], preds_high.shape[1]
                        ),
                        preds_high,
                    )
                )

            # Plot validation data versus predicted values
            output_names = [
                "Volume (m$^3$)",
                "Major Axis Length (m)",
                "Minor Axis Length (m)",
            ]
            fig, axes = plt.subplots(
                1, len(output_names), figsize=(16, 5), sharex=False, sharey=False
            )
            for j, ax in enumerate(axes):
                # Extract true values and predicted values
                y_true = targets_real_all[:, j]
                y_pred = preds_real_all[:, j]
                # Error bars: std = preds_high - preds_real (same as preds_real - preds_low)
                yerr = np.vstack(
                    (
                        preds_real_all[:, j] - preds_low_all[:, j],
                        preds_high_all[:, j] - preds_real_all[:, j],
                    )
                )
                # Scatter with error bars
                ax.errorbar(
                    y_true,
                    y_pred,
                    yerr=yerr,
                    fmt="o",
                    alpha=0.7,
                    ecolor="gray",
                    capsize=3,
                )
                # Line y=x
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], "r--")
                ax.set_title(output_names[j])
                ax.set_xlabel("Truth")
                ax.set_ylabel("Predicted")
                ax.set_aspect(1)
                ax.grid(True)

            plt.tight_layout()
            plt.savefig("validation.png")
            plt.close()

        # Plot loss graph
        plt.plot(train_loss, label="Loss (Train)")
        plt.plot(test_loss, label="Loss (Test)")
        plt.legend()
        plt.savefig("loss.png")
