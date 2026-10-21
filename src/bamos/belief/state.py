"""BeliefState bundle (spec 3)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bamos.belief.location_belief import LocationBelief
from bamos.mapping.grid import VoxelGrid


@dataclass
class BeliefState:
    grid: VoxelGrid  # the lattice the target-location candidates are drawn from (spec 5)
    location: LocationBelief
    step: int = 0  # number of completed observations t
    visited_pose_ids: set = field(default_factory=set)
    version: int = 0

    def candidate_centers(self) -> np.ndarray:
        return self.grid.centers(self.grid.unravel(self.location.candidate_voxels))

    def map_center(self) -> np.ndarray:
        return self.candidate_centers()[self.location.map_index()]
