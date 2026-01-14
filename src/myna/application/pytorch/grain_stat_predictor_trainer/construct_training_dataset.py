import zipfile
import fnmatch
import os
import io
import glob
import tempfile
import numpy as np
import polars as pl
import shutil
import torch
from pathlib import Path
from typing import List, Tuple, NamedTuple


class ArchiveFileDescriptor(NamedTuple):
    """Describes a file that is potentially located inside of an archive. The
    full path of the file is described as the concatenation of archive + path"""

    archive: str | None
    path: str


class TrainingDataPair(NamedTuple):
    """Describes a pair of input data and output data for training a model"""

    inputs: list[ArchiveFileDescriptor]
    outputs: list[ArchiveFileDescriptor]


def get_matching_filepaths(
    base_path: str | Path, pattern: str
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
    parent_directory: str | Path,
    case_dir_pattern: str,
    case_input_pattern: str,
    case_output_pattern: str,
) -> List[Tuple[Path, Path]]:
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
    cases = get_matching_filepaths(parent_directory, case_dir_pattern)
    for case in cases:
        input_files = get_matching_filepaths(case.path, case_input_pattern)
        output_files = get_matching_filepaths(case.path, case_output_pattern)
        data_pairs.append(TrainingDataPair(input_files, output_files))
    for data_pair in data_pairs:
        print("Inputs:")
        for input in data_pair.inputs:
            print(f"- {input}")
        print("Outputs:")
        for output in data_pair.outputs:
            print(f"- {output}")
    return data_pairs


def extract_zip_and_find_csvs(
    zip_filepath: Path, filename_prefix: str
) -> Tuple[str, List[str]]:
    # Make container for matching files
    matching_csv_files = []
    # Extract zipfile into temporary path
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        # Extract all files first
        zip_ref.extractall(temp_dir)
        # Find matching CSV files within the extracted directory
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.startswith(filename_prefix) and file.lower().endswith(".csv"):
                    matching_csv_files.append(os.path.join(root, file))

    # Sort the files to ensure consistent order (e.g., surfaces_thermal_1.csv, surfaces_thermal_2.csv, etc.)
    matching_csv_files.sort(
        key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x))))
    )
    # Return temporary directory and csv files
    return temp_dir, matching_csv_files


def create_grid_from_sparse_csv(
    csv_filepath: str,
    placeholder_value: float = np.nan,
    bounds: Tuple[float, float, float, float] = None,
) -> np.ndarray:
    """Creates a grid from a CSV with sparse grid points on a regular grid"""

    # Read CSV file
    data = np.loadtxt(csv_filepath, delimiter=",", skiprows=1)

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
    num_nans_channel = np.isnan(channel_data).sum()
    if num_nans_channel > 0:
        print(
            f"Warning: Channel data from {csv_filepath} contains {num_nans_channel} NaN values."
        )

    # Create the dense grid, initialized with the placeholder value
    grid = np.full((num_channels, height, width), placeholder_value, dtype=np.float32)

    # Populate the grid using the calculated integer indices
    grid[:, y_indices, x_indices] = channel_data.T

    # Report any nan values in grid data
    num_nans = np.isnan(grid).sum()
    if num_nans > 0:
        print(
            f"Warning: Created grid from {csv_filepath} contains {num_nans} NaN values."
        )

    return grid


def make_input_data(zip_filepath: Path) -> np.ndarray:
    """Generate input data for the neural network"""

    input_data = None
    temp_dir = None
    try:
        # Unzip files and add CSV data to a 3D array
        temp_dir, matching_csv_files = extract_zip_and_find_csvs(
            zip_filepath=zip_filepath, filename_prefix="surfaces_thermal_"
        )

        # For each csv, add it to 3D array
        input_data = make_input_data_from_surface_csv(matching_csv_files)

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return input_data


def make_input_data_from_surface_csv(
    matching_csv_files: List[str], compute_bounds=False
) -> np.ndarray:
    """Generate input data for the neural network from a list of csv files"""
    # Compute bounds from files
    min_x, min_y, max_x, max_y = (1e10, 1e10, -1e10, -1e10)
    bounds = None
    if compute_bounds:
        print("  - Computing lower bounds from CSV files")
        for csv_filepath in matching_csv_files:
            # Read CSV file
            data = np.loadtxt(csv_filepath, delimiter=",", skiprows=1)
            x_col, y_col = (0, 1)
            x_coords_float = data[:, x_col]
            y_coords_float = data[:, y_col]
            unique_x = np.unique(x_coords_float)
            unique_y = np.unique(y_coords_float)
            min_x = min(min_x, unique_x[0])
            min_y = min(min_y, unique_y[0])
            max_x = max(max_x, unique_x[-1])
            max_y = max(max_y, unique_y[-1])
        bounds = (min_x, min_y, max_x, max_y)

    # For each csv, add it to 3D array
    input_data = None
    for i, csv_file in enumerate(matching_csv_files):
        arr_2d = create_grid_from_sparse_csv(csv_file, bounds=bounds)
        # If no data, set the shape based on current array
        if input_data is None:
            input_data = np.empty(shape=arr_2d.shape + (len(matching_csv_files),))
        # Set data of array
        input_data[:, :, :, i] = arr_2d
    return input_data


def make_input_data_from_thesis_csv(
    matching_csv_files: List[str], compute_bounds=False
) -> np.ndarray:
    """Generate input data for the neural network from a list of csv files"""

    # Create surface files from thesis files if needed
    surface_csv_files = []
    min_x, min_y, max_x, max_y = (1e10, 1e10, -1e10, -1e10)
    bounds = None
    for thesis_datafile in matching_csv_files:
        extracted_file, xy_bounds = extract_surface(thesis_datafile)
        surface_csv_files.append(extracted_file)
        if compute_bounds and xy_bounds:
            min_x = min(min_x, xy_bounds[0])
            min_y = min(min_y, xy_bounds[1])
            max_x = max(max_x, xy_bounds[2])
            max_y = max(max_y, xy_bounds[3])
            bounds = (min_x, min_y, max_x, max_y)

    # For each csv, add it to 3D array
    input_data = None
    for i, csv_file in enumerate(surface_csv_files):
        arr_2d = create_grid_from_sparse_csv(csv_file, bounds=bounds)
        # If no data, set the shape based on current array
        if input_data is None:
            input_data = np.empty(shape=arr_2d.shape + (len(surface_csv_files),))
        # Set data of array
        input_data[:, :, :, i] = arr_2d
    return input_data


def make_output_data(csv_filepath: Path) -> np.ndarray:
    # Read csv and load as output
    file_data = np.genfromtxt(
        csv_filepath, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    # Get row for mean statistic
    mean_row_index = np.where(file_data["Statistic"] == "mean")[0][0]
    # Read in desired data
    mean_volume = file_data["Volume_m3"][mean_row_index]
    major_axis = file_data["Major_Axis_Length_m"][mean_row_index]
    minor_axis = file_data["Minor_Axis_Length_m"][mean_row_index]
    # Make into numpy array
    output_data = np.array([mean_volume, major_axis, minor_axis])
    return output_data


def make_dataset(data_pairs: List[Tuple[Path, Path]], dataset_path: Path = None):

    # Generate new data and save it
    print("- Generating new dataset from source folders...")

    # Loop over data pairs and extract data
    all_inputs, all_outputs = [], []
    # Changes:
    # zip_path -> input_path
    # csv_path -> output_path
    for i, (input_path, output_path) in enumerate(data_pairs):
        # Make input and output data
        input_data = make_input_data(input_path)
        output_data = make_output_data(output_path)
        # Append the numpy arrays to our lists
        all_inputs.append(input_data)
        all_outputs.append(output_data)

    # Stack arrays
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


def extract_surface(
    thesis_datafile: Path, top_suffix="_top", bot_suffix="_bot"
) -> Tuple[Path, Tuple[float, float, float, float]]:
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

    # Total path for source csv file
    if not thesis_datafile.exists():
        print(f"Warning: Thesis solidification file not found: {thesis_datafile}")
        return

    # Total path for output csv file
    output_file = thesis_datafile.parent / "surfaces_thermal_input_vectors.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Return early if output file already exists
    if output_file.exists():
        # Get x and y bounds of the data
        final_df = pl.read_csv(output_file)
        min_x = final_df["x"].min()
        max_x = final_df["x"].max()
        min_y = final_df["y"].min()
        max_y = final_df["y"].max()
        bounds = (min_x, min_y, max_x, max_y)
        return (output_file, bounds)

    # Read in the full CSV file
    df = pl.read_csv(thesis_datafile)

    # Define columns
    cols = ["x", "y", "G", "V", "depth", "numMelt"]  # initial columns in thesis data
    both_cols = ["G", "V", "numMelt"]  # columns to extract for both top and bottom
    final_cols = [
        "x",
        "y",
        "G_top",
        "V_top",
        "depth",
        "numMelt_top",
        "G_bot",
        "V_bot",
        "numMelt_bot",
    ]  # final columns in output data

    # Make top dataframe and add suffix
    top_df = df.filter(pl.col("z") == pl.col("z").max())
    top_df = top_df.select(cols).with_columns(
        [pl.col(c).alias(f"{c}{top_suffix}") for c in both_cols]
    )

    # Extract the bottom surface
    bottom_df = (
        df.group_by(["x", "y"])
        .agg(pl.col("z").arg_min().alias("idx_bottom"))
        .join(df.with_row_index(), left_on="idx_bottom", right_on="index")
        .select(cols)
    )
    bottom_df = bottom_df.select(cols).with_columns(
        [pl.col(c).alias(f"{c}{bot_suffix}") for c in both_cols]
    )

    # Merge top and bottom dataframes
    merged_df = top_df.join(bottom_df, on=["x", "y"], how="inner")

    # Drop and rename columns
    final_df = merged_df.select(final_cols)

    # Save the combined data
    final_df.write_csv(output_file)

    # Get x and y bounds of the data
    min_x = final_df["x"].min()
    max_x = final_df["x"].max()
    min_y = final_df["y"].min()
    max_y = final_df["y"].max()
    bounds = (min_x, min_y, max_x, max_y)

    return (output_file, bounds)


if __name__ == "__main__":
    find_data_pairs(
        "Z:/ConceptLaserM2-ORNL1/2024/01/2024-01-26 M2_AMMT_DOE_10/Myna/cases_ammt_doe_10",
        case_dir_pattern="P*/R*",
        case_input_pattern="Layer_*.zip/surfaces_thermal_FULL*.csv",
        case_output_pattern="grain_analysis.csv",
    )
