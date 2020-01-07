import pandas as pd

import seistech_internal as si

# Data locations
site_source_ffp = "/Users/Clus/code/work/nesi/nobackup/nesi00213/seistech/site_source/19p9/flt_site_source.db"
site_params_ffp = "/Users/Clus/code/work/NN_GMM/data/site_params/v18p6.csv"
source_params_ffp = "/Users/Clus/code/work/NN_GMM/data/src_params/cybershake_v19p5.csv"
im_db_ffp = "/Users/Clus/code/work/NN_GMM/data/im_data/cybershake_v19p5.h5"

# Load site_source data
# This is saved at a per station level in the db
cur_site = "CCCC"
with si.dbs.SiteSourceDB(site_source_ffp) as site_source_db:
    cur_site_df = site_source_db.station_data(cur_site)

# Load Site params
site_df = pd.read_csv(site_params_ffp)

# Load source params
source_df = pd.read_csv(source_params_ffp)

# Load IM data
# Saved at a per fault level
cur_fault = "Woodville"

with pd.HDFStore(im_db_ffp, "r") as store:
    cur_im_df = store[cur_fault]

exit()




