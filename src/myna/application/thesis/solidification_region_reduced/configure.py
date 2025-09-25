#
# Copyright (c) 2024 Oak Ridge National Laboratory.
#
# This file is part of Myna. For details, see the top-level license
# at https://github.com/ORNL-MDF/Myna/LICENSE.md.
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause.
#
import mistlib as mist
import os
from myna.core.workflow.load_input import load_input
from myna.application.thesis.parse import adjust_parameter, load_file_lines
import argparse
import sys
import shutil
import numpy as np
import polars as pl

from myna.application.thesis import Thesis


def configure_case(app, case_dir, res, myna_input="myna_data.yaml"):
    # Load input file
    input_path = os.path.join(case_dir, myna_input)
    settings = load_input(input_path)

    # Get part and layer info
    part = list(settings["build"]["parts"].keys())[0]
    region = list(settings["build"]["parts"][part]["regions"].keys())[0]
    layer = list(
        settings["build"]["parts"][part]["regions"][region]["layer_data"].keys()
    )[0]
    x_center = settings["build"]["parts"][part]["regions"][region]["x"]
    y_center = settings["build"]["parts"][part]["regions"][region]["y"]

    # Copy template to case directory, using user specified template if given
    app.set_template_path("thesis", "solidification_region_reduced", "template")
    app.copy(case_dir)

    # Set up scan path
    myna_scanfile = settings["build"]["parts"][part]["regions"][region]["layer_data"][
        layer
    ]["scanpath"]["file_local"]
    case_scanfile = os.path.join(case_dir, "Path.txt")
    shutil.copy(myna_scanfile, case_scanfile)
    df = pl.read_csv(case_scanfile, separator="\t")
    df = df.with_columns((pl.col("Z(mm)") * 0.0).alias("Z(mm)"))
    df.write_csv(case_scanfile, separator="\t")

    # Set beam data
    beam_file = os.path.join(case_dir, "Beam.txt")
    power = settings["build"]["parts"][part]["laser_power"]["value"]
    spot_size = settings["build"]["parts"][part]["spot_size"]["value"]
    spot_unit = settings["build"]["parts"][part]["spot_size"]["unit"]
    spot_scale = 1
    if spot_unit == "mm":
        spot_scale = 1e-3
    elif spot_unit == "um":
        spot_scale = 1e-6

    # For setting spot size, assume provided spot size is $D4 \sigma$
    # 3DThesis spot size is $\sqrt(6) \sigma$
    adjust_parameter(beam_file, "Width_X", 0.25 * np.sqrt(6) * spot_size * spot_scale)
    adjust_parameter(beam_file, "Width_Y", 0.25 * np.sqrt(6) * spot_size * spot_scale)
    adjust_parameter(beam_file, "Power", power)

    # Set up material properties
    material = settings["build"]["build_data"]["material"]["value"]
    material_dir = os.path.join(os.environ["MYNA_INSTALL_PATH"], "mist_material_data")
    try:
        mistPath = os.path.join(material_dir, f"{material}.json")
        mistMat = mist.core.MaterialInformation(mistPath)
        mistMat.write_3dthesis_input(os.path.join(case_dir, "Material.txt"))
        laser_absorption = mistMat.get_property("laser_absorption", None, None)
        adjust_parameter(beam_file, "Efficiency", laser_absorption)
    except:
        raise Exception(f'Material "{material}" not found in mist material database.')

    # Set preheat temperature
    preheat = settings["build"]["build_data"]["preheat"]["value"]
    adjust_parameter(os.path.join(case_dir, "Material.txt"), "T_0", preheat)

    # Update domain resolution and bounds
    domain_file = os.path.join(case_dir, "Domain.txt")
    adjust_parameter(domain_file, "Res", res)
    if app.args.xspan is not None:
        adjust_parameter(domain_file, "XMin", x_center - 0.5 * app.args.xspan)
        adjust_parameter(domain_file, "XMax", x_center + 0.5 * app.args.xspan)
    if app.args.yspan is not None:
        adjust_parameter(domain_file, "YMin", y_center - 0.5 * app.args.yspan)
        adjust_parameter(domain_file, "YMax", y_center + 0.5 * app.args.yspan)

    # Fix template Domain keys ("incorrect" to avoid issues with updating input file
    # programmatically with duplicate keys)
    file_lines = load_file_lines(domain_file)
    for old, new in zip(["XMin", "XMax"], ["Min", "Max"]):
        if app.args.xspan is None:
            file_lines = ["" if old in x else x for x in file_lines]
        else:
            file_lines = [x.replace(old, new) for x in file_lines]
    for old, new in zip(["YMin", "YMax"], ["Min", "Max"]):
        if app.args.yspan is None:
            file_lines = ["" if old in x else x for x in file_lines]
        else:
            file_lines = [x.replace(old, new) for x in file_lines]
    with open(domain_file, mode="w", encoding="utf-8") as f:
        file_contents = "\n".join(file_lines)
        f.write(file_contents)
        f.truncate()
    return


def main():

    app = Thesis("solidification_region_reduced")

    # Add arguments
    app.parser.add_argument(
        "--xspan",
        default=None,
        type=float,
        help="(float) width of region in X",
    )
    app.parser.add_argument(
        "--yspan",
        default=None,
        type=float,
        help="(float) width of region in Y",
    )
    app.args, _ = app.parser.parse_known_args()
    app.set_procs()

    # Get expected Myna output files
    step_name = os.environ["MYNA_STEP_NAME"]
    myna_files = app.settings["data"]["output_paths"][app.step_name]

    # Run each case
    for case_dir in [os.path.dirname(x) for x in myna_files]:
        configure_case(app, case_dir, app.args.res)


if __name__ == "__main__":
    main()
