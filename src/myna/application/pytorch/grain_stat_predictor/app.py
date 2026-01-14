import os
import glob
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myna.core.app import MynaApp
from ..grain_stat_predictor_trainer.app import GrainStatPredictorTrainerApp
from ..grain_stat_predictor_trainer.neural_net.architecture import Periodic3DCNN
from ..grain_stat_predictor_trainer.neural_net.normalizer import DatasetNormalizer
from ..grain_stat_predictor_trainer.construct_training_dataset import (
    make_input_data_from_thesis_csv,
)
from myna.application.pytorch import extract_patches_as_strided
from myna.core.utils import working_directory


class GrainStatPredictorApp(MynaApp):

    def __init__(self, name="grain_stat_predictor"):
        super().__init__(name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def parse_execute_arguments(self):
        """Check for arguments relevant to the configure step and update app settings"""
        # Parse app-specific arguments
        self.parser.add_argument(
            "--trained-model-dir",
            default=None,
            type=str,
            help="Directory containing the trained model output"
            "from grain_stat_predictor_trainer",
        )
        # TODO: Remove this field, as it should be derived from the Myna input file
        self.parser.add_argument(
            "--thermal-data-dir",
            default=None,
            type=str,
            help="Directory containing the thermal data files for prediction",
        )
        self.parser.add_argument(
            "--layer-start",
            default=None,
            type=int,
            help="Starting layer for prediction (inclusive)",
        )
        self.parser.add_argument(
            "--layer-stop",
            default=None,
            type=int,
            help="Stopping layer for prediction (inclusive), must >= layer start + 3",
        )
        self.parser.add_argument(
            "--batch-size",
            default=16,
            type=int,
            help="Batch size for neural network evaluation",
        )
        self.args, _ = self.parser.parse_known_args()
        self.mpiargs_to_current()

        # Update derived parameters
        self.set_procs()
        self.set_template_path("pytorch", "grain_stat_predictor")

    def execute(self):
        """Configure the training data for the model"""

        # Parse arguments
        self.parse_execute_arguments()

        # Create trainer app instance to get access to file paths
        trainer_app = GrainStatPredictorTrainerApp()

        # Load model
        print("- Loading trained model")
        with working_directory(self.args.trained_model_dir):
            with torch.serialization.safe_globals(
                [
                    torch.utils.data.dataset.Subset,
                    torch.utils.data.dataset.TensorDataset,
                    torch.nn.modules.activation.SiLU,
                    DatasetNormalizer,
                ]
            ):
                # Load the model args, state, and normalizer
                model_arg_dict = torch.load(trainer_app.model_arg_dict)
                trained_model_state_dict = torch.load(
                    trainer_app.trained_model_state_dict
                )
                normalizer = torch.load(trainer_app.data_file_normalizer)

                # Initialize the model and load the state dict
                model_silu = Periodic3DCNN(**model_arg_dict).to(self.device)
                model_silu.load_state_dict(trained_model_state_dict)
                model_silu.eval()  # Set to evaluation mode

        # Get input data, filtering data files based on specified layers
        print(f"- Getting input data from {self.args.thermal_data_dir}")
        # TODO: These files should be passed from a previous Myna step that outputs solidification data
        #    e.g. from 3DThesis, assumed to have the columns:
        #       - x: location in meters
        #       - y: location in meters
        #       - z: location in meters
        #       - G: temperature gradient during solidification in K/m
        #       - V: solidification velocity in m/s
        #       - depth: depth of the melt pool in meters
        #       - numMelt: number of times the material has melted
        csv_files = sorted(
            glob.glob(
                f"{self.args.thermal_data_dir.strip("\'\"")}/*/3dthesis_full/Data/*.Final.csv"
            )
        )
        for csv_file in csv_files:
            print("  - Found CSV file:", csv_file)
        layer_ids = [int(f.split(os.path.sep)[-4]) for f in csv_files]
        filtered_files = [
            Path(path)
            for layer_id, path in zip(layer_ids, csv_files)
            if self.args.layer_start <= layer_id <= self.args.layer_stop
        ]
        input_data_array = make_input_data_from_thesis_csv(
            filtered_files, compute_bounds=True
        )
        input_data_array = np.array(
            [input_data_array]
        )  # Add expected batch dimension, since currently only expecting 1 dataset (i.e. set of 4 files)
        input_tensor = torch.from_numpy(input_data_array).float()

        # Normalize input data, temporarily setting output channels to empty as they
        # are not known here
        orig_output_linear_channels = normalizer.output_linear_channels
        orig_output_log_channels = normalizer.output_log_channels
        normalizer.output_linear_channels = []
        normalizer.output_log_channels = []
        print(f"- {normalizer.input_linear_channels=}")
        print(f"- {normalizer.input_log_channels=}")
        norm_inputs_tensor, _ = normalizer.transform(
            inputs=input_tensor, outputs=input_tensor
        )

        # Remove batch dimension
        norm_inputs_tensor = norm_inputs_tensor[0]

        # Restore original normalizer settings
        normalizer.output_linear_channels = orig_output_linear_channels
        normalizer.output_log_channels = orig_output_log_channels

        # Save image of input tensor
        self.plot_image(
            norm_inputs_tensor.permute(1, 2, 0, 3).numpy()[:, :, :, 0],
            ["G_top", "V_top", "depth", "numMelt_top", "G_bot", "V_bot", "numMelt_bot"],
            "input_tensor.png",
        )

        # Extract patches from the input images
        print("- Constructing patches")
        # Reshape Tensor (C, H, W, D) -> (C, D, H, W) and extract patches
        # TODO: The neural network should be trained with the (C, D, H, W) format to
        #       avoid having to permute here and elsewhere in the model functions.
        #       Currently, the permute in the model forward function is commented out.
        PATCH_SIZE = 121
        STRIDE = 100
        norm_inputs_tensor = norm_inputs_tensor.permute(0, 3, 1, 2)
        input_patches, n_h, n_w = extract_patches_as_strided(
            norm_inputs_tensor, patch_size=PATCH_SIZE, stride=STRIDE
        )

        # Use DataLoader for batching
        patch_loader = DataLoader(
            input_patches, batch_size=self.args.batch_size, shuffle=False
        )

        # Evalute model on each patch
        print("- Evaluating model on patches and recombining")
        with torch.no_grad():
            # Get predictions
            preds_real_all = np.array([])
            mask = []
            for batch_inputs in patch_loader:

                # Determine if patch has too many nans and mask if so
                for batch_input in batch_inputs:
                    if torch.isnan(batch_input).sum() > (
                        0.25 * PATCH_SIZE * PATCH_SIZE
                    ):
                        mask.append(False)
                    else:
                        mask.append(True)

                # Get model prediction
                mu, log_var, _ = model_silu(batch_inputs)

                # Convert both prediction and target back to their original "real" scale
                mu_cpu, std_cpu = mu.cpu(), torch.exp(0.5 * log_var.cpu())

                # Un-normalize output data
                preds_real = normalizer.inverse_transform_outputs(mu_cpu).numpy()

                # Append for all batches
                preds_real_all = np.vstack(
                    (
                        preds_real_all.reshape(
                            preds_real_all.shape[0], preds_real.shape[1]
                        ),
                        preds_real,
                    )
                )

            # Recombine patches by assigning to a new empty grid
            pred_img = np.empty((n_h, n_w, preds_real_all.shape[1]))
            for i in range(n_h):
                for j in range(n_w):
                    patch_idx = i * n_w + j
                    pred_img[i, j, :] = preds_real_all[patch_idx]
                    # Mask out invalid patches
                    if not mask[patch_idx]:
                        pred_img[i, j, :] = (
                            np.ones_like(preds_real_all[patch_idx]) * np.nan
                        )
            output_names = [
                "Volume (m$^3$)",
                "Major Axis Length (m)",
                "Minor Axis Length (m)",
            ]
            self.plot_image(pred_img, output_names, "output_tensor.png")

    def plot_image(self, img: np.ndarray, var_names: list[str], export_file: str):
        """Plot the image (ni, nj, nvars) with one subplot per variable"""
        fig, axs = plt.subplots(
            1,
            len(var_names),
            figsize=(5 * len(var_names), 5),
            sharex=False,
            sharey=False,
        )
        for j, ax in enumerate(axs):
            # Plot the image
            im = ax.imshow(img[:, :, j], cmap="viridis", origin="lower")
            ax.set_aspect(1)
            ax.axis("off")

            # Add a horizontal colorbar on top of the image with the variable name
            divider1 = make_axes_locatable(ax)
            cax1 = divider1.append_axes("bottom", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax1, orientation="horizontal", label=var_names[j])

        # Save image
        plt.tight_layout(rect=[0.05, 0.15, 0.95, 0.95])
        plt.savefig(export_file, dpi=300)
        plt.close()
