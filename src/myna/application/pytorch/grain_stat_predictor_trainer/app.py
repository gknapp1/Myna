# Base imports
import glob
import zipfile
import fnmatch
from pathlib import Path
from typing import NamedTuple
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

from myna.core.app import MynaApp
from .construct_training_dataset import make_dataset
from .neural_net.architecture import Periodic3DCNN
from .neural_net.trainer import train_model
from .neural_net.normalizer import DatasetNormalizer


class ArchiveFileDescriptor(NamedTuple):
    """Describes a file that is potentially located inside of an archive. The
    full path of the file is described as the concatenation of archive + path"""

    archive: str | None
    path: str


class TrainingDataPair(NamedTuple):
    """Describes a pair of input data and output data for training a model"""

    inputs: list[ArchiveFileDescriptor]
    outputs: list[ArchiveFileDescriptor]


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
            "--case-dir-pattern",
            default="./*/",
            type=str,
            help="Pattern to match for finding case directories",
        )
        self.parser.add_argument(
            "--case-input-pattern",
            default="surfaces_thermal_input_vectors.csv",
            type=str,
            help="Pattern to match for finding inputs within each case directory",
        )
        self.parser.add_argument(
            "--case-output-pattern",
            default="grain_analysis.csv",
            type=str,
            help="Pattern to match for finding outputs within each case directory",
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
        dataset_path = Path(self.dataset_file).expanduser()

        if dataset_path.exists():
            # Load and return existing data
            print(f"- Loading dataset from {dataset_path}")
            data_dict = torch.load(dataset_path)
            inputs_tensor = data_dict["inputs"]
            outputs_tensor = data_dict["outputs"]
        else:
            # Find data pairs matching the specified patterns (one pair per case)
            data_pairs = self.find_data_pairs(
                base_directory,
                self.args.case_dir_pattern,
                self.args.case_input_pattern,
                self.args.case_output_pattern,
            )
            print(f"- Found {len(data_pairs)} training data pairs.")

            # TODO: Assemble dataset from the data pairs
            # - Need to extract files from zip to disk or to memory
            # - Might need to convert 3DThesis data to the expected format
            print("- Assembling dataset")
            inputs_tensor, outputs_tensor = make_dataset(
                data_pairs, dataset_path=dataset_path
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
        # and validation sets. Save tensors on CPU to avoid CUDA-device
        # specific pickles which may not load if CUDA is unavailable.
        full_dataset = TensorDataset(
            norm_inputs_tensor.cpu(), norm_outputs_tensor.cpu()
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
            # Use lists to collect per-batch arrays, then stack once to avoid
            # repeated reshapes/vstack edge-cases for the first batch.
            targets_list = []
            preds_real_list = []
            preds_low_list = []
            preds_high_list = []
            for batch_inputs, batch_targets in val_loader:
                # Move inputs to device for model inference
                batch_inputs = batch_inputs.to(self.device)

                # Run the model
                mu, log_var, _ = model_silu(batch_inputs)

                # Convert both prediction and target back to their original "real" scale
                mu_cpu, std_cpu = mu.cpu(), torch.exp(0.5 * log_var.cpu())

                # Make low, actual, and high predictions (+/- one std)
                preds_low = normalizer.inverse_transform_outputs(
                    mu_cpu - std_cpu
                ).numpy()
                preds_real = normalizer.inverse_transform_outputs(mu_cpu).numpy()
                preds_high = normalizer.inverse_transform_outputs(
                    mu_cpu + std_cpu
                ).numpy()

                # Get targets (already CPU from DataLoader since datasets are saved on CPU)
                targets_norm = batch_targets
                targets_real = normalizer.inverse_transform_outputs(
                    targets_norm
                ).numpy()

                # Append batch results
                targets_list.append(targets_real)
                preds_real_list.append(preds_real)
                preds_low_list.append(preds_low)
                preds_high_list.append(preds_high)

            # Stack lists into arrays (handle empty case gracefully)
            if len(targets_list) > 0:
                targets_real_all = np.vstack(targets_list)
                preds_real_all = np.vstack(preds_real_list)
                preds_low_all = np.vstack(preds_low_list)
                preds_high_all = np.vstack(preds_high_list)
            else:
                targets_real_all = np.empty((0, 3))
                preds_real_all = np.empty((0, 3))
                preds_low_all = np.empty((0, 3))
                preds_high_all = np.empty((0, 3))

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

    def get_matching_filepaths(
        self, base_path: str | Path, pattern: str
    ) -> list[ArchiveFileDescriptor]:
        """Get matching filepaths within the base paths, supporting searching for file paths
        within .zip archives.
        """
        path_parts = (Path(base_path) / Path(pattern)).parts
        is_zip = [True if ".zip" in x else False for x in path_parts]
        matches = []
        if any(is_zip):
            zip_dir_pattern = str(Path(*path_parts[: is_zip.index(True) + 1]))
            file_pattern = str(Path(*path_parts[is_zip.index(True) + 1 :]))
            zip_dirs = sorted(glob.glob(str(zip_dir_pattern)))
            for zip_dir in zip_dirs:
                print(f"- {zip_dir=}")
                with zipfile.ZipFile(zip_dir, mode="r") as zf:
                    # Get list of zipinfo objects that match the pattern in the .zip directory
                    zipinfos = [
                        x
                        for x in zf.infolist()
                        if fnmatch.fnmatch(x.filename, file_pattern) and not x.is_dir()
                    ]

                    # Record each file descriptor in the archive
                    for zi in zipinfos:
                        matches.append(ArchiveFileDescriptor(zip_dir, zi.filename))
        else:
            matches.extend(
                [
                    ArchiveFileDescriptor(None, x)
                    for x in sorted(glob.glob(str(Path(base_path) / Path(pattern))))
                ]
            )
        return matches

    def find_data_pairs(
        self,
        parent_directory: str | Path,
        case_dir_pattern: str,
        case_input_pattern: str,
        case_output_pattern: str,
    ) -> list[tuple[Path, Path]]:
        """
        Recursively finds pairs of (zip_file, csv_file) within a parent directory.

        A valid pair is found in a directory that contains:
        1. A file named exactly 'grain_analysis.csv'.
        2. Exactly one file ending with '.zip'.

        Args:
            parent_directory: The top-level directory to start the search from.

        Returns:
            A list of tuples, where each tuple is (path_to_zip, path_to_csv).
        """
        data_pairs = []
        cases = self.get_matching_filepaths(parent_directory, case_dir_pattern)
        for case in cases:
            input_files = self.get_matching_filepaths(case.path, case_input_pattern)
            output_files = self.get_matching_filepaths(case.path, case_output_pattern)
            data_pairs.append(TrainingDataPair(input_files, output_files))
        return data_pairs
