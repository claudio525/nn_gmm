from enum import StrEnum
from pathlib import Path
import numpy as np
import pandas as pd

import oq_wrapper as oqw

class ObsDataSource(StrEnum):
    NZGMDB = "NZGMDB"
    NGAWest2 = "NGAWest2"
    NGASubduction = "NGASubduction"

class NZGMDBVersion(StrEnum):
    v4p3_final = "v4.3_final"

class TectonicType(StrEnum):
    CRUSTAL = "crustal"
    SUBDUCTION_INTERFACE = "subduction_interface"
    SUBDUCTION_SLAB = "subduction_slab"
    OUTER_RISE = "outer_rise"
    MANTLE = "mantle"
    UNKNOWN = "unknown"




GMM_MAPPING = {
    oqw.constants.TectType.ACTIVE_SHALLOW: oqw.constants.GMM.Br_13,
    oqw.constants.TectType.VOLCANIC: oqw.constants.GMM.Br_13,
    oqw.constants.TectType.SUBDUCTION_INTERFACE: oqw.constants.GMM.K_20,
}

TECT_TYPES = [
    oqw.constants.TectType.SUBDUCTION_INTERFACE,
    oqw.constants.TectType.ACTIVE_SHALLOW,
    oqw.constants.TectType.VOLCANIC,
]

TECTONIC_TYPE_MAPPING = {
    "SUBDUCTION_INTERFACE": oqw.constants.TectType.SUBDUCTION_INTERFACE,
    "SUBDUCTION_SLAB": oqw.constants.TectType.SUBDUCTION_SLAB,
    "ACTIVE_SHALLOW": oqw.constants.TectType.ACTIVE_SHALLOW,
    "VOLCANIC": oqw.constants.TectType.VOLCANIC,
}
REVERSE_TECTONIC_TYPE_MAPPING = {v: k for k, v in TECTONIC_TYPE_MAPPING.items()}

PSA_KEYS = [
    "pSA_0.01",
    "pSA_0.02",
    "pSA_0.03",
    "pSA_0.04",
    "pSA_0.05",
    "pSA_0.075",
    "pSA_0.1",
    "pSA_0.12",
    "pSA_0.15",
    "pSA_0.17",
    "pSA_0.2",
    "pSA_0.25",
    "pSA_0.3",
    "pSA_0.4",
    "pSA_0.5",
    "pSA_0.6",
    "pSA_0.7",
    "pSA_0.75",
    "pSA_0.8",
    "pSA_0.9",
    "pSA_1.0",
    # "pSA_1.25",
    "pSA_1.5",
    "pSA_2.0",
    "pSA_2.5",
    "pSA_3.0",
    "pSA_4.0",
    "pSA_5.0",
    "pSA_6.0",
    "pSA_7.5",
    "pSA_10.0",
]
PSA_PERIODS = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
    0.12,
    0.15,
    0.17,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.9,
    1.0,
    # 1.25,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    7.5,
    10.0,
]

NON_PSA_IMS = ["PGA", "PGV", "CAV", "AI", "Ds575", "Ds595", "MMI"]
IMS = PSA_KEYS + NON_PSA_IMS

DB_PSA_KEYS = [
    "pSA_0p01",
    "pSA_0p02",
    "pSA_0p03",
    "pSA_0p04",
    "pSA_0p05",
    "pSA_0p075",
    "pSA_0p1",
    "pSA_0p12",
    "pSA_0p15",
    "pSA_0p17",
    "pSA_0p2",
    "pSA_0p25",
    "pSA_0p3",
    "pSA_0p4",
    "pSA_0p5",
    "pSA_0p6",
    "pSA_0p7",
    "pSA_0p75",
    "pSA_0p8",
    "pSA_0p9",
    "pSA_1p0",
    "pSA_1p25",
    "pSA_1p5",
    "pSA_2p0",
    "pSA_2p5",
    "pSA_3p0",
    "pSA_4p0",
    "pSA_5p0",
    "pSA_6p0",
    "pSA_7p5",
    "pSA_10p0",
]
PSA_KEYS_TO_DB = dict(zip(PSA_KEYS, DB_PSA_KEYS))
PSA_KEYS_TO_DB_SERIES = pd.Series(PSA_KEYS_TO_DB)
DB_PSA_KEYS_TO_PSA = dict(zip(DB_PSA_KEYS, PSA_KEYS))

DB_IM_KEYS = DB_PSA_KEYS + NON_PSA_IMS
DB_IMS_TO_IMS = dict(zip(DB_IM_KEYS, IMS))
IMS_TO_DB_IMS = dict(zip(IMS, DB_IM_KEYS))
IMS_TO_DB_IMS_SERIES = pd.Series(IMS_TO_DB_IMS)


GMM_PSA_MEAN_KEYS = [f"{k}_mean" for k in PSA_KEYS]
GMM_PSA_TOTAL_STD_KEYS = [f"{k}_std" for k in PSA_KEYS]

DB_GMM_PSA_MEAN_KEYS = [f"{k}_mean" for k in DB_PSA_KEYS]
DB_GMM_PSA_TOTAL_STD_KEYS = [f"{k}_std" for k in DB_PSA_KEYS]


# PRED_PSA_KEYS = [f"{k}_pred" for k in PSA_KEYS]
# PRED_NON_PSA_IMS = [f"{k}_pred" for k in NON_PSA_IMS]
# PRED_IMS = PRED_PSA_KEYS + PRED_NON_PSA_IMS

IM_SET_MAPPING = {"pSA": PSA_KEYS, "all": IMS}

MIN_MAX_PRE_PROCESS_CONFIG = {
    "magnitude": (2, 9),
    "rake": (-180, 180),
    "dip": (0, 90),
    "dtop": (0, 15),
    "dbottom": (5, 40),
    "vs30": (100, 1500),
    "z1p0": (0, 1.5),
    "z2p5": (0, 12.5),
    "lon": (166, 179),
    "lat": (-47.5, -34),
    "nztm_x": (952959.37518638, 2154468.7808133774),
    "nztm_y": (4721798.07382891, 6215675.54086617),
}

# Sample weighting
# MAG_WEIGHTING_BINS = np.arange(5.5, 8.5, 0.25)
MAG_WEIGHTING_BINS = np.array([5.5, 6.5, 7.25, 8.5])
MAG_WEIGHTING_BIN_NAMES = [
    f"{MAG_WEIGHTING_BINS[i]:.2f}_{MAG_WEIGHTING_BINS[i + 1]:.2f}"
    for i in range(len(MAG_WEIGHTING_BINS) - 1)
]
# RRUP_WEIGHTING_BINS = np.linspace(0, 300, 25)
RRUP_WEIGHTING_BINS = np.asarray([0, 30, 100, 300])
RRUP_WEIGHTING_BIN_NAMES = [
    f"{RRUP_WEIGHTING_BINS[i]:.2f}_{RRUP_WEIGHTING_BINS[i + 1]:.2f}"
    for i in range(len(RRUP_WEIGHTING_BINS) - 1)
]
# VS30_WEIGHTING_BINS = np.logspace(np.log(100), np.log(1200), 10, base=np.e)
VS30_WEIGHTING_BINS = np.asarray([0, 180, 360, 760, 1500])
VS30_WEIGHTING_BIN_NAMES = [
    f"{VS30_WEIGHTING_BINS[i]:.2f}_{VS30_WEIGHTING_BINS[i + 1]:.2f}"
    for i in range(len(VS30_WEIGHTING_BINS) - 1)
]


# NZ bounding box
NZ_BOUNDING_BOX = [166, 179, -47.5, -34.0]

COOK_STRAIT_REGION = [172.639, 176.35, -42.427, -40.475]
WELLINGTON_REGION = [174.74, 175, -41.44, -41.18]

BASIN_BOUNDARIES_DIR = Path(__file__).parent / "resources/basin_boundaries"