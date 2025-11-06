# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
from enum import Enum
import itertools
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class SimulationDataType(Enum):
    H5COMPACT = 1
    XDMF = 2
    SOURCE_ONLY = 3
    
class DataInterface(ABC):
    # Required attributes that concrete classes must define
    simulationDataType: SimulationDataType
    datasets: list[h5py.File]
    mesh: np.ndarray
    tags_field: list[str]
    tag_ufield: str
    tt: np.ndarray
    data_prune: int
    N: int
    u_shape: np.ndarray | list[int]
    tsteps: np.ndarray

    @property
    @abstractmethod
    def P(self) -> int:
        """Total number of time/space points."""
        pass

    @abstractmethod
    def u_pressures(self, idx: int) -> np.ndarray:
        """Get normalized u pressures for a given dataset index."""
        pass

def _calculate_u_pressure_minmax(
    datasets: list, tag_ufield: str, max_samples: int = 500
) -> tuple[float, float]:
    """Calculate min and max pressure values from datasets.

    Args:
        datasets: List of datasets (h5py.File or mock objects)
        tag_ufield: Tag/key for accessing pressure data in datasets
        max_samples: Maximum number of samples to use for estimation (default: 500)

    Returns:
        Tuple of (p_min, p_max) as floats
    """
    num_samples = min(max_samples, len(datasets))
    p_min_vals = []
    p_max_vals = []

    print(f"Estimating pressure min/max from {num_samples} samples...")
    for i in range(num_samples):
        upressures = datasets[i][tag_ufield][:]
        p_min_vals.append(np.min(upressures))
        p_max_vals.append(np.max(upressures))

    p_min = float(np.min(p_min_vals))
    p_max = float(np.max(p_max_vals))
    print(f"Pressure range: [{p_min:.4f}, {p_max:.4f}]")

    return p_min, p_max


def _normalize_spatial(data: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """Normalize spatial coordinates to [-1, 1] range.

    Args:
        data: Data to normalize
        xmin: Minimum value for normalization
        xmax: Maximum value for normalization

    Returns:
        Normalized data in [-1, 1] range
    """
    return 2 * (data - xmin) / (xmax - xmin) - 1


def _normalize_temporal(data: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """Normalize temporal coordinates.

    Args:
        data: Data to normalize
        xmin: Minimum spatial value for normalization
        xmax: Maximum spatial value for normalization

    Returns:
        Normalized temporal data
    """
    return data / (xmax - xmin) / 2


def _denormalize_spatial(data: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """Denormalize spatial coordinates from [-1, 1] range.

    Args:
        data: Normalized data to denormalize
        xmin: Minimum value for denormalization
        xmax: Maximum value for denormalization

    Returns:
        Denormalized spatial data
    """
    return (data + 1) / 2 * (xmax - xmin) + xmin


def _denormalize_temporal(data: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """Denormalize temporal coordinates.

    Args:
        data: Normalized data to denormalize
        xmin: Minimum spatial value for denormalization
        xmax: Maximum spatial value for denormalization

    Returns:
        Denormalized temporal data
    """
    return data * 2 * (xmax - xmin)

def paths_to_file_type(path, filetype, exclude=""):
    paths_to_file = []
    parent_dir_list = [
        i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))
    ]
    if len(parent_dir_list) == 0:
        parent_dir_list = [path]  # no subfolders
    for parent_dir_name in parent_dir_list:
        data_dir = os.path.join(path, parent_dir_name)
        for filename in os.listdir(data_dir):
            if filename.endswith(filetype) and (
                exclude == "" or filename.find(exclude) == -1
            ):
                paths_to_file.append(os.path.join(data_dir, filename))

    paths_to_file.sort()

    return paths_to_file

def get_nearest_from_coordinates(grid, coords):
    r0 = []
    r0_indxs = []

    for i_src in range(len(coords)):
        N_recvs = len(coords[i_src])
        coord_indxs = []

        for j_recv in range(N_recvs):
            indx = np.sum(np.abs(grid - coords[i_src][j_recv]), 1).argmin()
            coord_indxs.append(indx)

        r0.append(grid[coord_indxs].tolist())
        r0_indxs.append(coord_indxs)

    return np.array(r0), np.array(r0_indxs)

class DataH5Compact(DataInterface):
    simulationDataType: SimulationDataType = SimulationDataType.H5COMPACT

    datasets: list[h5py.File]
    mesh: np.ndarray
    u_shape: np.ndarray
    tsteps: np.ndarray
    tt: np.ndarray
    tags_field: list[str]
    tag_ufield: str
    data_prune: int
    N: int

    xmin: float
    xmax: float
    normalize_data: bool
    conn: np.ndarray

    # MAXNUM_DATASETS: SET TO E.G: 500 WHEN DEBUGGING ON MACHINES WITH LESS RESOURCES
    def __init__(
        self,
        data_path,
        tmax=float("inf"),
        t_norm=1,
        flatten_ic=True,
        data_prune=1,
        norm_data=False,
        MAXNUM_DATASETS=sys.maxsize,
        u_p_range=None,
    ):
        filenamesH5 = paths_to_file_type(data_path, ".h5", exclude="rectilinear")
        self.data_prune = data_prune
        self.normalize_data = norm_data

        # NOTE: we assume meshes, tags, etc are the same accross all xdmf datasets
        tag_mesh = "/mesh"
        tag_conn = "/conn"
        tag_umesh = "/umesh"
        tag_ushape = "umesh_shape"
        self.tags_field = ["/pressures"]
        self.tag_ufield = "/upressures"

        with h5py.File(filenamesH5[0]) as r:
            self.mesh = np.array(r[tag_mesh][:: self.data_prune])
            self.conn = (
                np.array(r[tag_conn])
                if self.data_prune == 1 and tag_conn in r
                else np.array([])
            )
            self.xmin, self.xmax = np.min(self.mesh), np.max(self.mesh)

            umesh_obj = r[tag_umesh]
            umesh = np.array(umesh_obj[:])
            self.u_shape = (
                np.array([len(umesh)], dtype=int)
                if flatten_ic
                else np.array(umesh_obj.attrs[tag_ushape][:], dtype=int)
            )
            self.tsteps = r[self.tags_field[0]].attrs["time_steps"]
            self.tsteps = np.array([t for t in self.tsteps if t <= tmax / t_norm])
            self.tsteps = (
                self.tsteps * t_norm
            )  # corresponding to c = 1 for same spatial / temporal resolution

            if self.normalize_data:
                self.mesh = self.normalize_spatial(self.mesh)
                # normalize relative to spatial dimension to keep ratio
                self.tsteps = self.normalize_temporal(self.tsteps)

        self.tt = np.repeat(self.tsteps, self.mesh.shape[0])
        self.N = len(filenamesH5)

        self.datasets = []
        for i in range(0, min(MAXNUM_DATASETS, len(filenamesH5))):
            filename = filenamesH5[i]
            if Path(filename).exists():
                # add file handles and keeps open
                self.datasets.append(h5py.File(filename, "r"))
            else:
                print(f"Could not be found (ignoring): {filename}")

        # Calculate min and max pressure values from sampled datasets
        if u_p_range is not None:
            self._u_p_min, self._u_p_max = u_p_range
            print(
                f"Using specified u pressure range: [{self._u_p_min:.4f}, {self._u_p_max:.4f}]"
            )
        else:
            self._u_p_min, self._u_p_max = _calculate_u_pressure_minmax(
                self.datasets, self.tag_ufield
            )

    # --- required abstract properties implemented ---
    @property
    def P_mesh(self):
        """Total number of mesh points."""
        return self.mesh.shape[0]

    @property
    def P(self):
        """Total number of time/space points."""
        return self.P_mesh * len(self.tsteps)

    @property
    def xxyyzztt(self):
        return np.hstack((self.xxyyzz, self.tt.reshape(-1, 1)))

    @property
    def xxyyzz(self):
        return np.tile(self.mesh, (len(self.tsteps), 1))

    def normalize_spatial(self, data):
        return _normalize_spatial(data, self.xmin, self.xmax)

    def normalize_temporal(self, data):
        return _normalize_temporal(data, self.xmin, self.xmax)

    def denormalize_spatial(self, data):
        return _denormalize_spatial(data, self.xmin, self.xmax)

    def denormalize_temporal(self, data):
        return _denormalize_temporal(data, self.xmin, self.xmax)

    def u_pressures(self, idx: int) -> np.ndarray:
        """Get normalized u pressures for a given dataset index."""
        dataset = self.datasets[idx]
        u_norm = _normalize_spatial(
            dataset[self.tag_ufield][:], self._u_p_min, self._u_p_max
        )
        return np.reshape(u_norm, self.u_shape)

    def __del__(self):
        for dataset in self.datasets:
            dataset.close()


class DatasetStreamer(Dataset):
    P_mesh: int
    P: int
    batch_size_coord: int
    data: DataInterface
    itercount: itertools.count
    __y_feat_extract_fn = Callable[[list], list]
    total_time = 0

    @property
    def N(self):
        return self.data.N

    @property
    def P_mesh(self):
        """Total number of mesh points."""
        return self.data.mesh.shape[0]

    @property
    def P(self):
        """Total number of time/space points."""
        return self.P_mesh * self.data.tsteps.shape[0]

    def __init__(self, data, batch_size_coord=-1, y_feat_extract_fn=None):
        # batch_size_coord: set to -1 if full dataset should be used (e.g. for validation data)
        self.data = data

        self.batch_size_coord = (
            batch_size_coord if batch_size_coord <= self.P else self.P
        )
        self.__y_feat_extract_fn = (
            (lambda y: y) if y_feat_extract_fn is None else y_feat_extract_fn
        )

        self.itercount = itertools.count()
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.data.datasets)

    def __getitem__(self, idx):
        dataset = self.data.datasets[idx]
        u = self.data.u_pressures(idx)

        if self.batch_size_coord > 0:
            indxs_coord = self.rng.choice(
                self.P, (self.batch_size_coord), replace=False
            )
        else:
            indxs_coord = np.arange(0, self.P)

        xxyyzz = self.data.mesh[np.mod(indxs_coord, self.data.P_mesh), :]
        tt = self.data.tt[indxs_coord].reshape(-1, 1)
        y = self.__y_feat_extract_fn(np.hstack((xxyyzz, tt)))

        # collect all field data for all timesteps - might be memory consuming
        # If memory load gets too heavy, consider selecting points at each timestep
        num_tsteps = len(self.data.tsteps)
        if self.data.simulationDataType == SimulationDataType.H5COMPACT:
            s = dataset[self.data.tags_field[0]][
                0:num_tsteps, :: self.data.data_prune
            ].flatten()[indxs_coord]
        elif self.data.simulationDataType == SimulationDataType.XDMF:
            s = np.empty((self.P), dtype=np.float32)
            for j in range(num_tsteps):
                s[j * self.data.P_mesh : (j + 1) * self.P_mesh] = dataset[
                    self.data.tags_field[j]
                ][:: self.data.data_prune]
            s = s[indxs_coord]
        elif self.data.simulationDataType == SimulationDataType.SOURCE_ONLY:
            s = []
        else:
            raise Exception(
                "Data format unknown: should be H5COMPACT, XDMF or SOURCE_ONLY"
            )

        # normalize
        x0 = (
            self.data.normalize_spatial(dataset["source_position"][:])
            if "source_position" in dataset
            else []
        )

        inputs = np.asarray(u), np.asarray(y)
        return inputs, np.asarray(s), indxs_coord, x0
    

def pytorch_collate(batch):
    """Collate function for PyTorch - converts batches to PyTorch tensors.

    Use this collator with PyTorch DataLoader for educational notebooks:
        DataLoader(dataset, batch_size=2, collate_fn=pytorch_collate)
    """

    if isinstance(batch[0], np.ndarray):
        return torch.from_numpy(np.stack(batch)).float()
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [pytorch_collate(samples) for samples in transposed]
    else:
        return torch.tensor(batch).float()