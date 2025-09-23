import shutil
from pathlib import Path
import logging
import torch
import pandas as pd
import numpy as np

import shapely
import geopandas as gpd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import optuna as opt

from qcore import coordinates
import ml_tools as mlt

from . import data
from . import constants
from . import utils
from . import preprocessing as pre


logger = logging.getLogger(__name__)


class LocationBatchData(data.BaseBatchData):

    def __init__(
        self,
        X: torch.Tensor,
        basin: torch.Tensor,
        district: torch.Tensor,
        authority: torch.Tensor,
    ):
        super().__init__()
        self.X = X
        self.y_basin = basin
        self.y_district = district
        self.y_authority = authority

    @property
    def n_samples(self):
        return self.X.shape[0]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"n_samples={self.n_samples.shape}, "
            f"X_shape={self.X.shape}, "
            f"y_basin_shape={self.y_basin.shape}, "
            f"y_district_shape={self.y_district.shape}, "
            f"y_authority_shape={self.y_authority.shape})"
        )


class LocationDataset(data.BaseDataset):

    def __init__(self, loc_df: pd.DataFrame, device: torch.device):
        super().__init__()
        self.loc_df = loc_df

        self.X_values = torch.from_numpy(self.loc_df[["nztm_x", "nztm_y"]].values).to(
            device, dtype=torch.float32
        )
        self.basin_values = torch.from_numpy(self.loc_df["basin"].values).to(
            device, dtype=torch.long
        )
        self.district_values = torch.from_numpy(self.loc_df["district"].values).to(
            device, dtype=torch.long
        )
        self.authority_values = torch.from_numpy(self.loc_df["authority"].values).to(
            device, dtype=torch.long
        )

    def __len__(self):
        return self.loc_df.shape[0]

    def get_batch(self, indices: np.ndarray | torch.Tensor) -> LocationBatchData:
        return LocationBatchData(
            X=self.X_values[indices],
            basin=self.basin_values[indices],
            district=self.district_values[indices],
            authority=self.authority_values[indices],
        )


class LocationNN(nn.Module):

    def __init__(
        self,
        n_inputs: int,
        n_basin_classes: int,
        n_district_classes: int,
        n_authority_classes: int,
        units: list[int],
        act_fn_str: str | None,
        embedding_dim: int,
        use_batch_norm: bool = False,
        dropout_rate: float | None = None,
    ):
        super().__init__()

        core_nn = nn.Sequential()
        for ix, cur_n_units in enumerate(units):
            core_nn.append(
                nn.Linear(
                    n_inputs if ix == 0 else units[ix - 1],
                    cur_n_units,
                    bias=True,
                )
            ),

            if use_batch_norm:
                core_nn.append(nn.BatchNorm1d(cur_n_units))

            if act_fn_str is not None:
                core_nn.append(mlt.torch.get_act_fn_layer(act_fn_str))

            if dropout_rate is not None and dropout_rate > 0:
                core_nn.append(nn.Dropout(dropout_rate))

        core_nn.append(nn.Linear(units[-1], embedding_dim))
        self.core_nn = core_nn

        self.basin_head = nn.Sequential(
            nn.Linear(units[-1], n_basin_classes),
        )

        self.district_head = nn.Sequential(
            nn.Linear(units[-1], n_district_classes),
        )

        self.authority_head = nn.Sequential(
            nn.Linear(units[-1], n_authority_classes),
        )

    def forward(self, X: torch.Tensor):
        features = self.core_nn(X)
        basin_out = self.basin_head(features)
        district_out = self.district_head(features)
        authority_out = self.authority_head(features)
        return basin_out, district_out, authority_out


def get_random_sites(n_sites: int):
    """Get random sites within New Zealand land area."""
    land_df = gpd.read_file(constants.NZ_LAND_SHAPEFILE)

    # Remove small islands
    land_proj = land_df.to_crs(epsg=2193)
    land_df["area_km2"] = land_proj.area / 1e6
    land_df = land_df[land_df["area_km2"] > 10.0]

    # Combine into a single polygon
    land_polygon = shapely.coverage_union_all(land_df.geometry)

    site_locs, ctr = [], 0
    while ctr < n_sites:
        cur_lon_values = np.random.uniform(
            constants.NZ_BOUNDING_BOX[0], constants.NZ_BOUNDING_BOX[1], 1000
        )
        cur_lat_values = np.random.uniform(
            constants.NZ_BOUNDING_BOX[2], constants.NZ_BOUNDING_BOX[3], 1000
        )

        mask = shapely.contains_xy(land_polygon, cur_lon_values, cur_lat_values)

        cur_sites = np.stack((cur_lon_values[mask], cur_lat_values[mask]), axis=1)

        site_locs.append(cur_sites)
        ctr += cur_sites.shape[0]

    site_locs = np.concatenate(site_locs, axis=0)
    site_locs = site_locs[:n_sites]

    return site_locs


def add_district_column(site_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a district column to the given site dataframe"""
    districts_df = gpd.read_file(constants.DISTRICT_SHAPEFILE)

    geo_site_df = gpd.GeoDataFrame(
        site_df,
        geometry=gpd.points_from_xy(site_df.lon, site_df.lat),
        crs=districts_df.crs,
    )

    joined = gpd.sjoin(geo_site_df, districts_df, how="left", predicate="within")
    joined = joined.rename(columns={"name": "district"}).drop(
        columns=["index_right", "id"]
    )
    joined = joined.loc[~joined.index.duplicated(keep="first")]

    site_df["district"] = joined["district"].values
    site_df = site_df.astype({"district": "category"})
    return site_df


def add_authority_column(site_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a territorial authority column to the given site dataframe"""
    assert (
        "nztm_x" in site_df.columns and "nztm_y" in site_df.columns
    ), "Site dataframe must contain 'nztm_x' and 'nztm_y' columns"

    auth_df = gpd.read_file(constants.AUTHORITY_SHAPEFILE)

    geo_site_df = gpd.GeoDataFrame(
        site_df,
        geometry=gpd.points_from_xy(site_df.nztm_x, site_df.nztm_y),
        crs=auth_df.crs,
    )

    joined = gpd.sjoin(geo_site_df, auth_df, how="left", predicate="within")
    joined = joined.drop(
        columns=[
            "index_right",
            "geometry",
            "LAND_AREA_",
            "AREA_SQ_KM",
            "SHAPE_Leng",
            "TA2025_V_1",
            "TA2025_V1_",
        ]
    ).rename(columns={"TA2025_V_2": "authority"})

    site_df["authority"] = joined["authority"].values
    site_df = site_df.astype({"authority": "category"})
    return site_df


def get_rand_site_df(
    n_sites: int,
    district_label_enc: LabelEncoder,
    authority_label_enc: LabelEncoder,
    basin_label_enc: LabelEncoder | None = None,
) -> pd.DataFrame:
    """
    Get a pre-processed DataFrame containing random
    sites within New Zealand land area.
    """
    site_locs = get_random_sites(n_sites=n_sites)

    site_df = pd.DataFrame(site_locs, columns=["lon", "lat"])
    nztm_coords = coordinates.wgs_depth_to_nztm(site_df[["lat", "lon"]].values)[:, ::-1]
    site_df["nztm_x"], site_df["nztm_y"] = nztm_coords[:, 0], nztm_coords[:, 1]

    site_df = utils.add_basin_column(site_df)
    site_df.loc[site_df.basin.isna(), "basin"] = "NiB"
    site_df = add_district_column(site_df)
    if (nan_mask := site_df.district.isna()).any():
        logger.debug(
            f"Dropping {nan_mask.sum()} sites not in any district out of {n_sites} total sites"
        )
        site_df = site_df.loc[~nan_mask]
    site_df = add_authority_column(site_df)
    if (nan_mask := site_df.authority.isna()).any():
        logger.debug(
            f"Dropping {nan_mask.sum()} sites not in any authority out of {n_sites} total sites"
        )
        site_df = site_df.loc[~nan_mask]

    pre_site_df = pre.preprocess_site_features(site_df, ["nztm_x", "nztm_y"])

    if basin_label_enc is None:
        basin_label_enc = LabelEncoder()
        basin_label_enc.fit(site_df["basin"])
    pre_site_df["basin"] = basin_label_enc.transform(site_df["basin"])

    pre_site_df["district"] = district_label_enc.transform(site_df["district"])

    pre_site_df["authority"] = authority_label_enc.transform(site_df["authority"])

    return (
        site_df,
        pre_site_df,
        basin_label_enc,
    )


def run_loc_model_training(
    n_train_sites: int,
    n_val_sites: int,
    n_epochs: int,
    units: list[int],
    l2_reg: float,
    batch_size: int,
    activation_fn: str,
    embedding_dim: int,
    dropout_rate: float,
    use_batch_norm: bool,
    device: torch.device,
    base_out_dir: Path,
    suffix: str = "",
    seed: int = 42,
    verbose: bool = True,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    auth_df = gpd.read_file(constants.AUTHORITY_SHAPEFILE)
    districts_df = gpd.read_file(constants.DISTRICT_SHAPEFILE)

    # Create label encoders for district and authority
    authorities = auth_df["TA2025_V_2"].unique().astype(str)
    districts = districts_df["name"].unique().astype(str)

    district_label_enc = LabelEncoder()
    district_label_enc.fit(districts)

    authority_label_enc = LabelEncoder()
    authority_label_enc.fit(authorities)
    assert authority_label_enc.classes_.size == authorities.size

    # Generate training and validation sites
    logger.info(f"Generating {n_train_sites} training sites")
    (
        train_site_df,
        train_pre_site_df,
        basin_label_enc,
    ) = get_rand_site_df(n_train_sites, district_label_enc, authority_label_enc)
    logger.info(f"Generating {n_val_sites} validation sites")
    val_site_df, val_pre_site_df, *_ = get_rand_site_df(
        n_val_sites,
        district_label_enc,
        authority_label_enc,
        basin_label_enc=basin_label_enc,
    )

    # Datasets and Dataloaders
    train_dataset = LocationDataset(train_pre_site_df, device=device)
    val_dataset = LocationDataset(val_pre_site_df, device=device)

    train_dataloader = data.CustomDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, use_torch=True, device=device
    )
    val_dataloader = data.CustomDataLoader(
        val_dataset, batch_size=1024, shuffle=False, use_torch=True, device=device
    )

    # Create the model
    model = LocationNN(
        n_inputs=2,
        n_basin_classes=len(basin_label_enc.classes_),
        n_district_classes=len(district_label_enc.classes_),
        n_authority_classes=len(authority_label_enc.classes_),
        units=units,
        act_fn_str=activation_fn,
        embedding_dim=embedding_dim,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=l2_reg)

    metrics = {
        "basin_loss_train": np.zeros(n_epochs),
        "basin_loss_val": np.zeros(n_epochs),
        "district_loss_train": np.zeros(n_epochs),
        "district_loss_val": np.zeros(n_epochs),
        "authority_loss_train": np.zeros(n_epochs),
        "authority_loss_val": np.zeros(n_epochs),
        "total_loss_train": np.zeros(n_epochs),
        "total_loss_val": np.zeros(n_epochs),
    }
    best_val_loss = np.inf
    best_model_epoch, best_model_state = None, None

    # Training loop
    for i in range(n_epochs):
        logger.debug(f"Epoch: {i + 1}/{n_epochs}")

        # Training
        n_samples = 0
        model.train()
        for cur_batch in tqdm(train_dataloader, disable=not verbose):
            optimizer.zero_grad()

            basin_logits, district_logits, authority_logits = model(cur_batch.X)

            basin_loss = F.cross_entropy(basin_logits, cur_batch.y_basin)
            district_loss = F.cross_entropy(district_logits, cur_batch.y_district)
            authority_loss = F.cross_entropy(authority_logits, cur_batch.y_authority)

            loss = basin_loss + district_loss + authority_loss
            loss.backward()
            optimizer.step()

            n_samples += cur_batch.n_samples

            metrics["basin_loss_train"][i] += basin_loss.item()
            metrics["district_loss_train"][i] += district_loss.item()
            metrics["authority_loss_train"][i] += authority_loss.item()
            metrics["total_loss_train"][i] += loss.item()

        metrics["basin_loss_train"][i] /= n_samples
        metrics["district_loss_train"][i] /= n_samples
        metrics["authority_loss_train"][i] /= n_samples
        metrics["total_loss_train"][i] /= n_samples

        ## Validation
        model.eval()
        n_samples = 0

        with torch.no_grad():
            for cur_batch in tqdm(val_dataloader, disable=not verbose):
                basin_logits, district_logits, authority_logits = model(cur_batch.X)

                basin_loss = F.cross_entropy(basin_logits, cur_batch.y_basin)
                district_loss = F.cross_entropy(district_logits, cur_batch.y_district)
                authority_loss = F.cross_entropy(
                    authority_logits, cur_batch.y_authority
                )

                loss = basin_loss + district_loss + authority_loss

                n_samples += cur_batch.n_samples

                metrics["basin_loss_val"][i] += basin_loss.item()
                metrics["district_loss_val"][i] += district_loss.item()
                metrics["authority_loss_val"][i] += authority_loss.item()
                metrics["total_loss_val"][i] += loss.item()

        metrics["basin_loss_val"][i] /= n_samples
        metrics["district_loss_val"][i] /= n_samples
        metrics["authority_loss_val"][i] /= n_samples
        metrics["total_loss_val"][i] /= n_samples

        # Keep track of the best model
        if metrics["total_loss_val"][i] < best_val_loss:
            best_model_epoch = i
            best_val_loss = metrics["total_loss_val"][i]
            best_model_state = model.state_dict()

        logger.info(f"Epoch {i + 1}/{n_epochs} completed.")
        logger.info(
            f"\n{'Training':<20} {'Validation':<20}\n"
            f"{'Loss:':<12} {metrics['total_loss_train'][i]:<12.6f} {'Loss:':<12} {metrics['total_loss_val'][i]:<12.6f}\n"
            f"{'  Basin:':<12} {metrics['basin_loss_train'][i]:<12.6f} {'  Basin:':<12} {metrics['basin_loss_val'][i]:<12.6f}\n"
            f"{'  District:':<12} {metrics['district_loss_train'][i]:<12.6f} {'  District:':<12} {metrics['district_loss_val'][i]:<12.6f}\n"
            f"{'  Authority:':<12} {metrics['authority_loss_train'][i]:<12.6f} {'  Authority:':<12} {metrics['authority_loss_val'][i]:<12.6f}"
        )

    # Load the best model
    model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        train_basin_logits, train_district_logits, train_authority_logits = model(
            train_dataset.X_values
        )
        val_basin_logits, val_district_logits, val_authority_logits = model(
            val_dataset.X_values
        )

    train_basin_preds = torch.argmax(train_basin_logits, dim=1).cpu().numpy()
    train_district_preds = torch.argmax(train_district_logits, dim=1).cpu().numpy()
    train_authority_preds = torch.argmax(train_authority_logits, dim=1).cpu().numpy()
    val_basin_preds = torch.argmax(val_basin_logits, dim=1).cpu().numpy()
    val_district_preds = torch.argmax(val_district_logits, dim=1).cpu().numpy()
    val_authority_preds = torch.argmax(val_authority_logits, dim=1).cpu().numpy()

    # Get the accuracy on the training & validation set
    train_basin_acc = np.mean(
        train_basin_preds == train_dataset.basin_values.cpu().numpy()
    )
    train_district_acc = np.mean(
        train_district_preds == train_dataset.district_values.cpu().numpy()
    )
    train_authority_acc = np.mean(
        train_authority_preds == train_dataset.authority_values.cpu().numpy()
    )
    val_basin_acc = np.mean(val_basin_preds == val_dataset.basin_values.cpu().numpy())
    val_district_acc = np.mean(
        val_district_preds == val_dataset.district_values.cpu().numpy()
    )
    val_authority_acc = np.mean(
        val_authority_preds == val_dataset.authority_values.cpu().numpy()
    )

    logger.info(
        f"\n{'Training Acc':<20} {'Validation Acc':<20}\n"
        f"{'  Basin:':<12} {train_basin_acc:<12.6f} {'  Basin:':<12} {val_basin_acc:<12.6f}\n"
        f"{'  District:':<12} {train_district_acc:<12.6f} {'  District:':<12} {val_district_acc:<12.6f}\n"
        f"{'  Authority:':<12} {train_authority_acc:<12.6f} {'  Authority:':<12} {val_authority_acc:<12.6f}"
    )

    # Create output directory
    (outdir := base_out_dir / mlt.utils.create_run_name(suffix=suffix)).mkdir(
        parents=False, exist_ok=False
    )

    metadata = {
        "best_model_epoch": int(best_model_epoch),
        "best_val_loss": float(best_val_loss),
    }
    mlt.utils.write_to_yaml(
        metadata, outdir / "metadata.yaml")
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_parquet(outdir / "metrics.parquet")

    # Save the model
    torch.save(model, outdir / "loc_model.pt")

    metrics_keys = ["total_loss", "basin_loss", "district_loss", "authority_loss"]
    for cur_metric in metrics_keys:
        fig = mlt.plotting.plot_metrics(metrics, [cur_metric])

        fig.savefig(outdir / f"{cur_metric}_loc_nn.png", dpi=300)
        plt.close(fig)

    return outdir


def hp_objective(
    trial: opt.Trial,
    base_out_dir: Path,
    n_train_sites: int,
    n_val_sites: int,
    n_epochs: int,
    device: str,
) -> float:
    """
    Objective function for hyperparameter optimization using Optuna.
    Should not be used for anything else.
    """
    log_ffp = base_out_dir / f"trial_{trial.number:03d}.log"
    
    # Add file handler to logger
    file_handler = logging.FileHandler(log_ffp)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Starting trial {trial.number}")

    n_layers = trial.suggest_int("n_layers", 1, 4)
    unit_size = trial.suggest_categorical("unit_size", [16, 32, 64, 128])
    units = [unit_size] * n_layers

    l2_reg = trial.suggest_categorical(
        "l2_reg", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    )
    dropout_rate = trial.suggest_categorical(
        "dropout_rate", [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    )
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048])
    activation_fn = trial.suggest_categorical(
        "activation_fn", ["relu", "elu", "leaky_relu"]
    )
    use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])

    out_dir = run_loc_model_training(
        n_train_sites,
        n_val_sites,
        n_epochs,
        units,
        l2_reg,
        batch_size,
        activation_fn,
        dropout_rate,
        use_batch_norm,
        device,
        base_out_dir,
        suffix=f"trial_{trial.number:03d}",
        verbose=False,
    )

    # Remove file handler from logger
    logger.removeHandler(file_handler)
    file_handler.close()

    metadata = mlt.utils.load_yaml(out_dir / "metadata.yaml")
    best_val_loss, best_epoch = metadata["best_val_loss"], metadata["best_model_epoch"]
    trial.user_attrs["best_val_loss"] = best_val_loss
    trial.user_attrs["best_model_epoch"] = best_epoch

    shutil.move(log_ffp, out_dir / log_ffp.name)

    return best_val_loss
