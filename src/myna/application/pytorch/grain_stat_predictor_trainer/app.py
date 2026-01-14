# Base imports
import os
import shutil
import glob
import zipfile
import fnmatch
from pathlib import Path
from typing import NamedTuple
import numpy as np
import polars as pl
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

from myna.core.app import MynaApp
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


def load_archive_file_to_df(afd: ArchiveFileDescriptor) -> pl.DataFrame:
    if afd.archive is None:
        return pl.read_csv(afd.path)
    else:
        with zipfile.ZipFile(afd.archive) as zf:
            with zf.open(afd.path) as f:
                return pl.read_csv(f)


class GrainStatPredictorTrainerApp(MynaApp):
    """Application for training a neural network to predict grain statistics"""

    def __init__(self, name="grain_stat_predictor_trainer"):
        super().__init__(name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_file = "dataset.pt"
        self.data_file_normalized_training = "data_normalized_training.pt"
        self.data_file_normalizer = "normalizer.pt"
        self.model_arg_dict = "model_arg_dict.pt"
        self.trained_model_state_dict = "trained_model_state_dict.pt"
        self.trained_model_loss_dict = "trained_model_loss_dict.pt"
        self.input_surface_top_suffix = "_top"
        self.input_surface_bot_suffix = "_bot"
        self.expected_thesis_data_cols = ["x", "y", "G", "V", "depth", "numMelt"]
        self.expected_surface_data_cols = [
            "x",
            "y",
            f"G{self.input_surface_top_suffix}",
            f"V{self.input_surface_top_suffix}",
            "depth",
            f"numMelt{self.input_surface_top_suffix}",
            f"G{self.input_surface_bot_suffix}",
            f"V{self.input_surface_bot_suffix}",
            f"numMelt{self.input_surface_bot_suffix}",
        ]
        self.expect_output_cols = [
            "Volume (m^3) [mean]",
            "Equivalent Diameter (m) [mean]",
            "Major Axis Length (m) [mean]",
            "Minor Axis Length (m) [mean]",
            "Volume (m^3) [std]",
            "Equivalent Diameter (m) [std]",
            "Major Axis Length (m) [std]",
            "Minor Axis Length (m) [std]",
        ]
        self.layers_per_input = 4

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
        self.args, _ = self.parser.parse_known_args()
        self.mpiargs_to_current()

        # Update derived parameters
        self.set_procs()
        self.set_template_path("pytorch", "grain_stat_predictor_trainer")

    def parse_execute_arguments(self):
        """Check for arguments relevant to the configure step and update app settings"""
        # Parse app-specific arguments
        self.parser.add_argument(
            "--test-train-split",
            default=0.8,
            type=float,
            help="Fractional split of dataset for testing and training,"
            " for example, 0.8 = 80/20 split",
        )
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

            # Assemble dataset from data pairs
            print("- Assembling dataset")
            inputs_tensor, outputs_tensor = self.make_dataset(
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

        # Create the full dataset. Save tensors on CPU to avoid CUDA-device
        # specific pickles which may not load if CUDA is unavailable.
        full_dataset = TensorDataset(
            norm_inputs_tensor.cpu(), norm_outputs_tensor.cpu()
        )
        torch.save(full_dataset, self.data_file_normalized_training)

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
            full_dataset = torch.load(self.data_file_normalized_training)
            train_split = self.args.test_train_split
            train_size = int(train_split * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )
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
            "in_channels": len(self.expected_surface_data_cols) - 2,
            "out_features": len(self.expect_output_cols),
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
            fig, axes = plt.subplots(
                1,
                len(self.expect_output_cols),
                figsize=(16, 5),
                sharex=False,
                sharey=False,
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
                ax.set_title(self.expect_output_cols[j])
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
        """Get matching filepaths within the base paths, supporting searching for
        file paths within .zip archives.
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
    ) -> list[TrainingDataPair]:
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

    def make_dataset(
        self, data_pairs: list[TrainingDataPair], dataset_path: Path = None
    ):
        """Create dataset"""

        # Generate new data and save it
        print("- Generating new dataset from source folders...")

        # Loop over data pairs and extract data
        all_inputs, all_outputs = [], []
        for data_pair in data_pairs:
            # Make input and output data
            print("- Data pair:")
            print("  - Inputs:")
            for fp in data_pair.inputs:
                print(f"    - {fp}")
            print("  - Outputs:")
            for fp in data_pair.outputs:
                print(f"    - {fp}")
            if not self.valid_data_pair(data_pair):
                continue
            input_data = self.input_data_to_array(data_pair.inputs)
            output_data = self.output_data_to_array(data_pair.outputs)
            # Append the numpy arrays to our lists
            all_inputs.append(input_data)
            all_outputs.append(output_data)

        # Stack arrays
        # - Input shape: (N, C, D, H, W)
        final_inputs_np = np.stack(all_inputs, axis=0)
        final_outputs_np = np.stack(all_outputs, axis=0)

        # Convert the final numpy arrays into PyTorch tensors
        inputs_tensor = torch.from_numpy(final_inputs_np).float()
        outputs_tensor = torch.from_numpy(final_outputs_np).float()

        # If filepath isn't none, try saving
        if dataset_path is not None:
            print(f"- Saving dataset to {dataset_path}...")
            # Save the tensors to a single file using a dictionary
            dataset_dict = {"inputs": inputs_tensor, "outputs": outputs_tensor}
            torch.save(dataset_dict, dataset_path)

        return inputs_tensor, outputs_tensor

    def valid_data_pair(self, data_pair: TrainingDataPair) -> bool:
        """Boolean test for if data_pair is valid"""
        
        if len(data_pair.inputs) != self.layers_per_input:
            print(f"{data_pair.inputs=}")
            print(f"Expected {self.layers_per_input} input files for data pair, not {len(data_pair.inputs)}")
            return False
        if len(data_pair.outputs) != 1:
            print(f"{data_pair.outputs=}")
            print(f"Expected 1 output file for data pair, not {len(data_pair.outputs)}")
            return False
        return True

    def input_data_to_array(self, files: list[ArchiveFileDescriptor]) -> np.ndarray:
        """Generate input data for the neural network from a single data pair input list"""

        # Extract data from files and pass to formatting functions
        dfs = []
        filenames = []
        input_data = None
        for i, afd in enumerate(files):
            # Validate that there are the correct number of files
            # TODO: The number of layer files should probably be an input variable
            if len(files) != 4:
                continue

            filenames.append(afd.path)
            dfs.append(load_archive_file_to_df(afd))

        # Sort by filename, assuming filename corresponds with layer order
        dfs = [a for _, a in sorted(zip(filenames, dfs), key=lambda k: k[0])]
        filenames = sorted(filenames)

        # Determine if CSV is 3D solidification data or the surface extraction
        data_arrays = []
        for df, fn in zip(dfs, filenames):
            # Handle converting 3DThesis data to surface data
            if all([x in df.columns for x in self.expected_thesis_data_cols]):
                data_arrays.append(self.extract_surface(df.select(self.expected_thesis_data_cols)))
            # Handle data that is already in the surface format
            elif all([x in df.columns for x in self.expected_surface_data_cols]):
                data_arrays.append(df.select(self.expected_surface_data_cols))
            # Handle data that is in the incorrect format
            else:
                print(f"{fn=}")
                print(f"{df.columns=}")
                print(f"{self.expected_surface_data_cols=}")
                print([x in df.columns for x in self.expected_surface_data_cols])
                error_msg = f"{fn} is not in the expected format for input data"
                raise LookupError(error_msg)

        # Stack the arrays
        input_data = None
        for i, data in enumerate(data_arrays):
            arr_2d = self.create_grid_from_sparse_df(data)
            # If no data, set the shape based on the first array
            # - Assumes that all data will have same shape
            if input_data is None:
                input_data = np.empty(shape=arr_2d.shape + (len(data_arrays),))
            # Set data of array (C, H, W, D)
            input_data[:, :, :, i] = arr_2d

        # Input data shape needs to be transposed (C, H, W, D) -> (C, D, H, W)
        # - C: Channel
        # - D: Depth, e.g., layers
        # - H: Y-dimension
        # - W: X-dimension
        input_data = np.transpose(input_data, [0, 3, 1, 2])
        return input_data

    def output_data_to_array(self, files: list[ArchiveFileDescriptor]) -> np.ndarray:
        """Convert the output data to an array"""
        # There should only be one datafile
        if len(files) != 1:
            error_msg = f"Expected 1 output datafile, not {len(files)}"
            raise ValueError(error_msg)

        # Load the datafile
        afd = files[0]
        df = load_archive_file_to_df(afd)

        # Assume a single-row table with table columns forming the output
        return df.select(self.expect_output_cols).to_numpy().flatten()

    def create_grid_from_sparse_df(
        self,
        df: pl.DataFrame,
        placeholder_value: float = np.nan,
        bounds: tuple[float, float, float, float] = None,
    ) -> np.ndarray:
        """Creates a grid from a polars DataFrame with sparse grid points on a regular grid
        
        The returned grid has a shape (C, H, W)"""

        data = df.to_numpy()

        # Define columns for x and y and read data
        x_col, y_col = (0, 1)
        x_coords_float = data[:, x_col]
        y_coords_float = data[:, y_col]

        # Find unique sorted coordinates to establish the grid axes
        unique_x = np.unique(x_coords_float)
        unique_y = np.unique(y_coords_float)

        # Determine grid parameters
        if bounds is None:
            min_x = unique_x[0]
            min_y = unique_y[0]
            max_x = unique_x[-1]
            max_y = unique_y[-1]
        else:
            min_x, min_y, max_x, max_y = bounds
        res_x = unique_x[1] - unique_x[0]
        res_y = unique_y[1] - unique_y[0]

        # # Grid dimensions are simply the number of unique points along each axis
        # width = len(unique_x)
        # height = len(unique_y)

        # Calculate grid dimensions based on bounds and resolution
        width = int(np.round((max_x - min_x) / res_x)) + 1
        height = int(np.round((max_y - min_y) / res_y)) + 1

        # Convert float coordinates to integer grid indices
        x_indices = np.rint((x_coords_float - min_x) / res_x).astype(int)
        y_indices = np.rint((y_coords_float - min_y) / res_y).astype(int)

        # Separate channel data and populate the grid
        num_total_cols = data.shape[1]
        all_col_indices = np.arange(num_total_cols)
        channel_col_indices = np.delete(all_col_indices, [x_col, y_col])
        channel_data = data[:, channel_col_indices]
        num_channels = channel_data.shape[1]

        # Report any nan values in channel data
        # TODO: Make sure that NaNs are handled in the normalizer and training in
        #       a reasonable way
        num_nans_channel = np.isnan(channel_data).sum()
        if num_nans_channel > 0:
            print(f"Warning: Channel data contains {num_nans_channel} NaN values.")

        # Create the dense grid, initialized with the placeholder value
        grid = np.full(
            (num_channels, height, width), placeholder_value, dtype=np.float32
        )

        # Populate the grid using the calculated integer indices
        grid[:, y_indices, x_indices] = channel_data.T

        # Report any nan values in grid data
        num_nans = np.isnan(grid).sum()
        if num_nans > 0:
            print(
                f"Warning: Created grid from data that contains {num_nans} NaN values."
            )

        return grid

    def extract_surface(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extracts the top and bottom surfaces from a 3DThesis output solidification data
        CSV file and saves them into a new CSV file that is compatible with ML input vector.

        Args:
            thesis_datafile: Path to the input 3DThesis solidification CSV file.
            top_suffix: Suffix to append to top surface columns.
            bot_suffix: Suffix to append to bottom surface columns.

        Returns:
            tuple:
            - Path to the output CSV file with extracted surfaces.
            - A tuple of (min_x, min_y, max_x, max_y) bounds of the data."""

        # Define columns that use _top and _bot suffix
        # (assume all colunmns with _top have corresponding _bot)
        both_cols = [
            x
            for x in self.expected_surface_data_cols
            if x.endswith(self.input_surface_top_suffix)
        ]

        # Make top dataframe and add suffix
        top_df = df.filter(pl.col("z") == pl.col("z").max())
        top_df = top_df.select(self.expected_thesis_data_cols).with_columns(
            [pl.col(c).alias(f"{c}{self.input_surface_top_suffix}") for c in both_cols]
        )

        # Extract the bottom surface
        bottom_df = (
            df.group_by(["x", "y"])
            .agg(pl.col("z").arg_min().alias("idx_bottom"))
            .join(df.with_row_index(), left_on="idx_bottom", right_on="index")
            .select(self.expected_thesis_data_cols)
        )
        bottom_df = bottom_df.select(self.expected_thesis_data_cols).with_columns(
            [pl.col(c).alias(f"{c}{self.input_surface_bot_suffix}") for c in both_cols]
        )

        # Merge top and bottom dataframes
        merged_df = top_df.join(bottom_df, on=["x", "y"], how="inner")

        # Return only expected columns
        return merged_df.select(self.expected_surface_data_cols)
