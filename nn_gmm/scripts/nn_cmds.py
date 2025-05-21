from pathlib import Path

import torch
import typer
from sklearn.model_selection import train_test_split

import nn_gmm as nng


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"



app = typer.Typer()


@app.command("train-gmm")
def train_gmm(run_config_ffp: Path):
    logger = nng.utils.setup_logging(Path("./nn_cmds.log"))

    run_config = nng.RunConfig.from_config_kwargs(
        config_ffp=run_config_ffp,
        device=device,
    )

    logger.info(f"Using device: {device.upper()}")

    # Get the data
    with nng.IMDB(run_config.imdb_ffp) as imdb:
        event_df = imdb.get_event_df()
        site_df = imdb.get_site_df(max_grid_level=0)

    events, sites = event_df.event_id.values.astype(str), site_df.site_id.values.astype(str)
    logger.info(f"Number of available events: {len(events)}")
    logger.info(f"Number of available sites: {len(sites)}")

    # Split into training and validation sets
    train_events, val_events = train_test_split(
        events, test_size=0.2, random_state=run_config.seed
    )
    train_sites, val_sites = train_test_split(
        sites, test_size=0.2, random_state=run_config.seed
    )
    logger.info(
        f"Number of events - Training: {len(train_events)}, Validation: {len(val_events)}"
    )
    logger.info(
        f"Number of sites - Training: {len(train_sites)}, Validation: {len(val_sites)}"
    )

    # Run model training
    nng.nn_gmm.run_model_training(
        run_config,
        event_df,
        site_df,
        train_events,
        val_events,
        train_sites,
        val_sites,
    )


@app.command("train-cv")
def train_cv(run_config_ffp: Path):
    pass


if __name__ == "__main__":
    app()
