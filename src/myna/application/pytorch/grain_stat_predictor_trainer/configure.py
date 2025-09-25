#
# Copyright (c) 2024 Oak Ridge National Laboratory.
#
# This file is part of Myna. For details, see the top-level license
# at https://github.com/ORNL-MDF/Myna/LICENSE.md.
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause.
#
"""Script to be executed by the configure stage of `myna.core.workflow.run` to set up
a valid pytorch training case based on the specified user inputs and template
"""
from myna.application.pytorch.grain_stat_predictor_trainer import (
    GrainStatPredictorTrainerApp,
)


def configure():
    """Configure all case directories"""
    app = GrainStatPredictorTrainerApp()
    app.configure()


if __name__ == "__main__":
    configure()
