"""Find DR ephys sessions on S3 with sync + NI-DAQ + sound stim files."""
import _creds  # noqa
import concurrent.futures as cf, json, re, s3fs

fs = s3fs.S3FileSystem()
ec = json.load(open("all_ecephys.json"))

def inspect(name):
    root = f"aind-ephys-data/{name}"
    try:
        beh = [p.split("/")[-1] for p in fs.ls(f"{root}/behavior")]
    except FileNotFoundError:
        return None
    stims = [b for b in beh if re.match(r"(DynamicRouting1|RFMapping)_\d+_\d+_\d+\.hdf5$", b)]
    syncs = [b for b in beh if re.match(r"\d{8}T\d{6}\.h5$", b) or b.endswith(".sync")]
    if not stims or len(syncs) != 1:
        return None
    try:
        nodes = [p for p in fs.ls(f"{root}/ecephys_clipped") if "Record Node" in p]
    except FileNotFoundError:
        return None
    rec_dirs = []
    for n in nodes:
        for cont in fs.glob(f"{n}/experiment*/recording*/continuous/NI-DAQmx-*"):
            rec_dirs.append(cont.split("/continuous/")[0])
    if not rec_dirs:
        return None
    return dict(session=name, stims=stims, sync=syncs[0],
                rec_dirs=sorted(set(rec_dirs)),
                nodes=[n.split("/")[-1] for n in nodes])

names = [n for n in ec if n.split("_")[2] >= "2022-06-01"]
out = []
with cf.ThreadPoolExecutor(16) as ex:
    for r in ex.map(inspect, names):
        if r:
            out.append(r)
out.sort(key=lambda r: r["session"])
json.dump(out, open("sessions.json", "w"), indent=1)
print(len(names), "checked;", len(out), "usable")
for r in out:
    print(r["session"], len(r["stims"]), len(r["rec_dirs"]), r["stims"])
