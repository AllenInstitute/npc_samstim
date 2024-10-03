from __future__ import annotations

import contextlib
import logging

import h5py
import npc_io
import npc_stim
import numpy as np
from DynamicRoutingTask.Analysis.DynamicRoutingAnalysisUtils import DynRoutData

logger = logging.getLogger(__name__)


def get_sam(
    stim_path_or_data: npc_io.PathLike | h5py.File,
) -> DynRoutData:
    stim_data = npc_stim.get_stim_data(stim_path_or_data)
    if not isinstance(stim_data, h5py.File):
        raise TypeError(f"Expected h5py.File, got {type(stim_data)}")
    obj = DynRoutData()
    try:
        obj.loadBehavData(filePath="dummy_366122_", h5pyFile=stim_data)
    except Exception as exc:
        raise TypeError(
            "Loading Sam's `DynRoutData` object requires data from a `DynamicRouting1` hdf5 file"
        ) from exc
    return obj


def is_opto(
    stim_path_or_data: npc_io.PathLike | h5py.File,
) -> bool:
    """
    >>> is_opto('s3://aind-ephys-data/ecephys_670248_2023-08-01_11-27-17/behavior/DynamicRouting1_670248_20230801_120304.hdf5')
    True
    """
    with contextlib.suppress(TypeError):
        return bool(
            (onset := getattr(get_sam(stim_path_or_data), "trialOptoOnsetFrame", None))
            is not None
            and np.any(~np.isnan(onset.squeeze()))
        )
    return False


def is_galvo_opto(
    stim_path_or_data: npc_io.PathLike | h5py.File,
) -> bool:
    """
    >>> is_galvo_opto('s3://aind-ephys-data/ecephys_670248_2023-08-01_11-27-17/behavior/DynamicRouting1_670248_20230801_120304.hdf5')
    True
    >>> is_galvo_opto('s3://aind-scratch-data/dynamic-routing/DynamicRoutingTask/Data/677352/DynamicRouting1_677352_20231013_155330.hdf5')
    False
    """
    with contextlib.suppress(TypeError, AttributeError):
        for param in (
            "trialGalvoX",
            "trialGalvoVoltage",
        ):  # trialGalvoVoltage is the original format, pre-March 2024
            if (
                voltage := getattr(get_sam(stim_path_or_data), param, None)
            ) is not None:
                return not all(np.isnan(a).any() for a in voltage)
    return False
