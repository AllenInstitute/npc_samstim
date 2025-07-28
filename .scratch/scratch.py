import pathlib

import npc_samstim

root = pathlib.Path(r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_636766_20230123")
npc_samstim.get_stim_latencies_from_nidaq_recording(
    stim_file_or_dataset=next(root.glob("DynamicRouting1_*.hdf5")),
    recording_dirs=(
            's3://aind-ephys-data/ecephys_636766_2023-01-23_00-00-00/ecephys_clipped/Record Node 110/experiment1/recording1',
            's3://aind-ephys-data/ecephys_636766_2023-01-23_00-00-00/ecephys_clipped/Record Node 112/experiment1/recording1',
    ),
    sync=next(root.glob("*.h5")),
    waveform_type='opto',
)