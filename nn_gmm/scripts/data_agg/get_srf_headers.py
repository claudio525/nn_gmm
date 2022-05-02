"""Script for retrieving the srf header data"""
import shutil
from pathlib import Path

import pandas as pd
from qcore import srf

# Cybershake v20p4
# sources_dir = Path("/isilon/cybershake/v20p4/Sources")
# out_srf_dir = Path("/home/cbs51/tmp/srf_files")
# out_ffp = Path("/home/cbs51/tmp/srf_headers.pickle")

# Small validation
# sources_dir = Path("/nesi/nobackup/nesi00213/RunFolder/Validation/v20p5p8/Data/Sources")
# out_srf_dir = Path("/home/cbs51/tmp/srf_data/small_val")
# out_ffp = Path("/home/cbs51/tmp/srf_data/small_val/srf_headers.pickle")

# Moderate validation
sources_dir = Path("/nesi/nobackup/nesi00213/RunFolder/Validation/v20p6p0/Data/Sources")
out_srf_dir = Path("/home/cbs51/tmp/srf_data/mod_val")
out_ffp = Path("/home/cbs51/tmp/srf_data/mod_val/srf_headers.pickle")


srf_ffps = list(sources_dir.glob("*/Srf/*.srf"))

result_dict = {}
for ix, cur_ffp in enumerate(srf_ffps):
    print(f"Processing {ix}/{len(srf_ffps)}")

    cur_srf_name = f"{cur_ffp.stem.split('_', maxsplit=1)[0]}.srf"
    if not (out_srf_dir / cur_srf_name).exists():
        shutil.copy(cur_ffp, out_srf_dir / cur_srf_name)

    result_dict[cur_ffp.stem] = srf.read_header(str(cur_ffp), idx=True)

pd.to_pickle(result_dict, out_ffp)