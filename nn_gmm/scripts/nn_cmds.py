from pathlib import Path
import logging

import torch
import typer

import nn_gmm as nng

logger = nng.utils.setup_logging(Path("./nn_cmds.log"))

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device.upper()}")

app = typer.Typer()

@app.command("train-gmm")
def train_gmm(run_config_ffp: Path):
    run_config = nng.RunConfig.from_config_kwargs(
        config_ffp=run_config_ffp,
        device=device,
    )

    # Get the data
    with nng.IMDB(run_config.imdb_ffp) as imdb:
        event_df =imdb.get_event_df()
        site_df = imdb.get_site_df(max_grid_level=0)

    print("wtf")


@app.command("train-cv")
def train_cv(run_config_ffp: Path):
    pass


if __name__ == "__main__":
    app()