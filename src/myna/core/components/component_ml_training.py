#
# Copyright (c) 2024 Oak Ridge National Laboratory.
#
# This file is part of Myna. For details, see the top-level license
# at https://github.com/ORNL-MDF/Myna/LICENSE.md.
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause.
#
"""Define Component subclasses for machine learning training"""

from myna.core.components.component import Component


class ComponentGrainStatPredictorTrainer(Component):
    """Component for grain statistic predictor training tasks"""

    def __init__(self):
        Component.__init__(self)
        self.input_requirement = None
        self.output_requirement = None


class ComponentGrainStatPredictor(Component):
    """Component for grain statistic prediction tasks"""

    def __init__(self):
        Component.__init__(self)
        self.input_requirement = None
        self.output_requirement = None
