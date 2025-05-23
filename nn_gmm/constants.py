import pandas as pd

PSA_KEYS = [
    'pSA_0.01', 'pSA_0.02', 'pSA_0.03', 'pSA_0.04', 'pSA_0.05', 'pSA_0.075',
    'pSA_0.1', 'pSA_0.12', 'pSA_0.15', 'pSA_0.17', 'pSA_0.2', 'pSA_0.25',
    'pSA_0.3', 'pSA_0.4', 'pSA_0.5', 'pSA_0.6', 'pSA_0.7', 'pSA_0.75',
    'pSA_0.8', 'pSA_0.9', 'pSA_1.0', 'pSA_1.25', 'pSA_1.5', 'pSA_2.0',
    'pSA_2.5', 'pSA_3.0', 'pSA_4.0', 'pSA_5.0', 'pSA_6.0', 'pSA_7.5',
    'pSA_10.0'
]
PSA_PERIODS = [
    0.01, 0.02, 0.03, 0.04, 0.05, 0.075,
    0.1, 0.12, 0.15, 0.17, 0.2, 0.25,
    0.3, 0.4, 0.5, 0.6, 0.7, 0.75,
    0.8, 0.9, 1.0, 1.25, 1.5, 2.0,
    2.5, 3.0, 4.0, 5.0, 6.0, 7.5,
    10.0
]
DB_PSA_KEYS = ["pSA_0p01", "pSA_0p02", "pSA_0p03", "pSA_0p04", "pSA_0p05", "pSA_0p075",
                "pSA_0p1", "pSA_0p12", "pSA_0p15", "pSA_0p17", "pSA_0p2", "pSA_0p25",
                "pSA_0p3", "pSA_0p4", "pSA_0p5", "pSA_0p6", "pSA_0p7", "pSA_0p75",
                "pSA_0p8", "pSA_0p9", "pSA_1p0", "pSA_1p25", "pSA_1p5", "pSA_2p0",
                "pSA_2p5", "pSA_3p0", "pSA_4p0", "pSA_5p0", "pSA_6p0", "pSA_7p5",
                "pSA_10p0"]
PSA_KEYS_TO_DB = dict(zip(PSA_KEYS, DB_PSA_KEYS))
PSA_KEYS_TO_DB_SERIES = pd.Series(PSA_KEYS_TO_DB)
DB_PSA_KEYS_TO_PSA = dict(zip(DB_PSA_KEYS, PSA_KEYS))

NON_PSA_IMS = ["PGA", "PGV", "CAV", "AI", "Ds575", "Ds595", "MMI"]
IMS = PSA_KEYS + NON_PSA_IMS

IM_SET_MAPPING = {
    "pSA": PSA_KEYS,
    "all": IMS
}

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
    "lat": (-47.5, -34.2)
}