import torch


class DatasetNormalizer:
    """
    A class to handle complex normalization for inputs and outputs.

    Learns statistics from the training data and applies them to any dataset.
    Handles mixed log/linear scaling and is robust to outliers using quantiles.
    """

    def __init__(
        self,
        input_linear_channels,
        input_log_channels,
        output_linear_channels,
        output_log_channels,
        quantile_clip=0.01,
    ):

        # Define which channel indices get which transformation
        self.input_linear_channels = input_linear_channels
        self.input_log_channels = input_log_channels
        self.output_linear_channels = output_linear_channels
        self.output_log_channels = output_log_channels

        # Clip at this quantile
        self.quantile_clip = quantile_clip

        # Learned parameters for all possible transformations
        self.input_linear_means, self.input_linear_stds = None, None
        self.input_log_means, self.input_log_stds = None, None
        self.output_linear_means, self.output_linear_stds = None, None
        self.output_log_means, self.output_log_stds = None, None

        # Set to be fit
        self.isFit = False

        # Set nan fill option
        self.nan_fill = None # TODO: allow setting in init

    @staticmethod
    def _nanstd(x: torch.Tensor, dim=None, keepdim=False, ddof=1):
        """
        Computes the standard deviation of a tensor over given dimensions, ignoring NaNs,
        with Bessel's correction (unbiased estimator).

        Args:
            x (torch.Tensor): The input tensor.
            dim (int or tuple, optional): The dimension or dimensions to reduce.
            keepdim (bool, optional): Whether the output tensor has `dim` retained or not.
            ddof (int, optional): Delta Degrees of Freedom. The divisor used in calculations is N - ddof.
        """
        # 1. Calculate the mean, ignoring NaNs
        mean = torch.nanmean(x, dim=dim, keepdim=True)

        # 2. Calculate the sum of squared differences
        sum_squared_diff = torch.nansum((x - mean) ** 2, dim=dim, keepdim=keepdim)

        # 3. Count the number of non-NaN elements
        n = torch.sum(~torch.isnan(x), dim=dim, keepdim=keepdim)

        # 4. Calculate the unbiased variance. Ensure the divisor is at least 1.
        # N - ddof can be 0 or negative if there are few data points, so clamp it.
        divisor = torch.clamp(n - ddof, min=1)
        variance = sum_squared_diff / divisor

        # 5. Handle cases with 0 or 1 non-NaN elements, where std is 0.
        variance[n <= ddof] = 0.0

        return torch.sqrt(variance)

    @staticmethod
    def _compute_multidim_quantiles(
        data: torch.Tensor, q_values: list, is_input_data: bool
    ):
        """
        Helper subroutine to compute quantiles over multiple dimensions
        by flattening the data, as torch.quantile only supports a single dim.
        """
        if is_input_data:
            # For inputs (N, C, D, H, W), we compute per-channel stats
            num_channels = data.shape[1]
            # Permute to (C, N, D, H, W) then reshape to (C, -1)
            permuted_data = data.permute(1, 0, 2, 3, 4)
            flattened_data = permuted_data.reshape(num_channels, -1)
            # Compute quantiles for each channel along the flattened dimension
            quantiles = torch.quantile(
                flattened_data, torch.tensor(q_values, device=data.device), dim=1
            )
            # Reshape results back to a broadcastable shape (q, C, 1, 1, 1)
            return quantiles.reshape(len(q_values), num_channels, 1, 1, 1)
        else:
            # For outputs (N, C), we can compute directly along the batch dim
            return torch.quantile(
                data, torch.tensor(q_values, device=data.device), dim=0
            )

    def fit(self, train_inputs: torch.Tensor, train_outputs: torch.Tensor):
        """Calculates and stores normalization parameters from the training data."""

        # --- Fit Input Parameters ---
        if self.input_log_channels:
            # Get log inputs and "log" them
            log_data_in = train_inputs[:, self.input_log_channels, ...]
            log_data_in = torch.log(log_data_in)

            # Clip outliers based on the entire distribution of log-scaled data
            q_low = torch.nanquantile(log_data_in, self.quantile_clip)
            q_high = torch.nanquantile(log_data_in, 1.0 - self.quantile_clip)
            clipped_log_data_in = torch.clamp(log_data_in, q_low, q_high)

            # Find means and stds
            self.input_log_means = torch.nanmean(
                clipped_log_data_in, dim=(0, 2, 3, 4), keepdim=True
            )
            self.input_log_stds = self._nanstd(
                clipped_log_data_in, dim=(0, 2, 3, 4), keepdim=True
            )

        if self.input_linear_channels:
            # Get linear inputs
            linear_data_in = train_inputs[:, self.input_linear_channels, ...]

            # Clip outliers based on the entire distribution of linear-scaled data
            q_low = torch.nanquantile(linear_data_in, self.quantile_clip)
            q_high = torch.nanquantile(linear_data_in, 1.0 - self.quantile_clip)
            clipped_linear_data_in = torch.clamp(linear_data_in, q_low, q_high)

            # Find means and stds
            self.input_linear_means = torch.nanmean(
                clipped_linear_data_in, dim=(0, 2, 3, 4), keepdim=True
            )
            self.input_linear_stds = self._nanstd(
                clipped_linear_data_in, dim=(0, 2, 3, 4), keepdim=True
            )

        # --- Fit Output Parameters ---
        if self.output_log_channels:
            # Get log outputs and "log" them
            log_data_out = train_outputs[:, self.output_log_channels]
            log_data_out = torch.log(log_data_out)

            # Clip outliers based on the entire distribution of log-scaled data
            q_low = torch.nanquantile(log_data_out, self.quantile_clip)
            q_high = torch.nanquantile(log_data_out, 1.0 - self.quantile_clip)
            clipped_log_data_out = torch.clamp(log_data_out, q_low, q_high)

            # Find means and stds
            self.output_log_means = torch.nanmean(
                clipped_log_data_out, dim=(0,), keepdim=True
            )
            self.output_log_stds = self._nanstd(
                clipped_log_data_out, dim=(0,), keepdim=True
            )

        if self.output_linear_channels:
            # Get log outputs and "log" them
            linear_data_out = train_outputs[:, self.output_linear_channels]
            linear_data_out = torch.log(log_data_out)

            # Clip outliers based on the entire distribution of log-scaled data
            q_low = torch.nanquantile(linear_data_out, self.quantile_clip)
            q_high = torch.nanquantile(linear_data_out, 1.0 - self.quantile_clip)
            clipped_linear_data_out = torch.clamp(linear_data_out, q_low, q_high)

            # Find means and stds
            self.output_linear_means = torch.nanmean(
                clipped_linear_data_out, dim=(0,), keepdim=True
            )
            self.output_linear_stds = self._nanstd(
                clipped_linear_data_out, dim=(0,), keepdim=True
            )

        self.isFit = True
        print("Normalizer fit to training data.")

    def transform(self, inputs: torch.Tensor, outputs: torch.Tensor):
        """Applies the stored normalization to a new dataset."""
        if not self.isFit:
            raise RuntimeError("Normalizer must be fit before transforming data.")

        # --- Transform inputs --- #
        norm_inputs = inputs.clone()
        # Apply log-scale and standardize
        if self.input_log_channels:
            log_part_in = torch.log(norm_inputs[:, self.input_log_channels, ...])
            norm_inputs[:, self.input_log_channels, ...] = (
                log_part_in - self.input_log_means
            ) / self.input_log_stds
        # Standardize linear
        if self.input_linear_channels:
            linear_part_in = norm_inputs[:, self.input_linear_channels, ...]
            norm_inputs[:, self.input_linear_channels, ...] = (
                linear_part_in - self.input_linear_means
            ) / self.input_linear_stds

        # --- Transform outputs --- #
        # Apply log-scale and standardize
        norm_outputs = outputs.clone()
        if self.output_log_channels:
            log_part_out = torch.log(norm_outputs[:, self.output_log_channels])
            norm_outputs[:, self.output_log_channels] = (
                log_part_out - self.output_log_means
            ) / self.output_log_stds
        # Standardize linear
        if self.output_linear_channels:
            linear_part_out = norm_outputs[:, self.output_linear_channels]
            norm_inputs[:, self.input_linear_channels] = (
                linear_part_out - self.output_linear_means
            ) / self.output_linear_stds

        # Fill NaNs if desired
        try:
            if self.nan_fill is not None:
                norm_inputs = torch.nan_to_num(norm_inputs, nan=self.nan_fill)
                norm_outputs = torch.nan_to_num(norm_outputs, nan=self.nan_fill)
        except Exception as e:
            # TODO: Remove this
            # Temporary behavior until model is retrained with new normalizer
            print(f"Warning: Could not apply NaN fill in normalizer: {e}")
            norm_inputs = torch.nan_to_num(norm_inputs, nan=0.0)
            norm_outputs = torch.nan_to_num(norm_outputs, nan=0.0)
        
        return norm_inputs, norm_outputs

    def inverse_transform_outputs(self, norm_outputs: torch.Tensor):
        """Converts normalized model predictions back to their original 'real' scale."""
        if not self.isFit:
            raise RuntimeError("Normalizer must be fit before using inverse transform.")

        # --- Transform outputs --- #
        # Apply log-scale and standardize
        real_outputs = norm_outputs.clone()
        if self.output_log_channels:
            log_part_out = real_outputs[:, self.output_log_channels]
            real_outputs[:, self.output_log_channels] = (
                log_part_out * self.output_log_stds
            ) + self.output_log_means
            real_outputs[:, self.output_log_channels] = torch.exp(real_outputs)

        # Standardize linear
        if self.output_linear_channels:
            linear_part_out = real_outputs[:, self.output_linear_channels]
            real_outputs[:, self.output_linear_channels] = (
                linear_part_out * self.output_linear_stds
            ) + self.output_linear_means

        return real_outputs
