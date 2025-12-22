"""Scripts for generating inputs for running DS simulation."""

import logging
import multiprocessing as mp
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

import typer

from source_modelling import sources, magnitude_scaling

from workflow.defaults import DefaultsVersion
from workflow.realisations import (
    Magnitudes,
    Rakes,
    RealisationMetadata,
    RupturePropagationConfig,
    SourceConfig,
    VelocityModelParameters,
)
from workflow.scripts.generate_velocity_model_parameters import (
    generate_velocity_model_parameters,
)


import nn_gmm as nng

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command("gen-realisations")
def gen_realisations(
    ds_sources_ffp: Path, mag_rrup_ffp: Path, output_dir: Path, n_procs: int = 1
) -> None:
    logger = nng.utils.setup_logging()

    source_df = pd.read_csv(ds_sources_ffp, index_col=0)

    if n_procs == 1:
        for _, cur_source in tqdm(source_df.iterrows(), total=len(source_df)):
            _process_source(cur_source, mag_rrup_ffp, output_dir, logger)
    else:
        mp_func = partial(
            _process_source,
            mag_rrup_ffp=mag_rrup_ffp,
            base_output_dir=output_dir,
            logger=logger,
        )
        with mp.Pool(n_procs) as pool:
            logger.info(f"Using {n_procs} processes to generate realisations.")
            list(
                tqdm(
                    pool.imap_unordered(
                        mp_func,
                        [row for _, row in source_df.iterrows()],
                    ),
                    total=len(source_df),
                )
            )


def _process_source(
    source_series: pd.Series,
    mag_rrup_ffp: Path,
    base_output_dir: Path,
    logger: logging.Logger,
) -> None:
    if (output_dir := base_output_dir / f"{source_series.name}").exists():
        logger.info(f"Realisation output {output_dir.name} config already exists, skipping")
        return
    output_dir.mkdir(exist_ok=False)
    rel_ffp = output_dir / "realisation.json"

    length_km, width_km = magnitude_scaling.magnitude_to_length_width(
        magnitude_scaling.ScalingRelation.CONTRERAS_SLAB2020,
        source_series.mag,
        source_series.rake,
    )
    length_m = length_km * 1000  # Convert km to meters
    width_m = width_km * 1000  # Convert km to meters

    source_geometry = sources.Point.from_lat_lon_depth(
        point_coordinates=np.asarray(
            (
                source_series["lat"],
                source_series["lon"],
                source_series["depth"] * 1000,
            )
        ),
        length_m=length_m,
        width_m=width_m,
        strike=source_series.strike,
        dip=source_series.dip,
        dip_dir=(source_series.strike + 90) % 360,
    )
    source_config = SourceConfig(
        source_geometries={str(source_series.name): source_geometry}
    )

    magnitudes = Magnitudes(
        magnitudes={str(source_series.name): float(source_series.mag)}
    )
    rakes = Rakes(rakes={str(source_series.name): float(source_series.rake)})

    vel_config = VelocityModelParameters.read_from_defaults(DefaultsVersion.v24_2_2_2)
    vel_config.version = "2.06"
    vel_config.topo_type = "BULLDOZED"
    vel_config.rrup_interpolants = pd.read_csv(mag_rrup_ffp).values.T

    rupture_config = RupturePropagationConfig(
        rupture_causality_tree={int(source_series.name): None},
        jump_points={},
        hypocentre=np.array([0.5, 0.5]),
    )

    metadata = RealisationMetadata(
        name=str(source_series.name),
        version="1",
        defaults_version=DefaultsVersion.v24_2_2_2,
        tag="NHM2010",
    )

    for config in [
        metadata,
        source_config,
        rupture_config,
        magnitudes,
        rakes,
        vel_config,
    ]:
        config.write_to_realisation(rel_ffp)

    try:
        generate_velocity_model_parameters(rel_ffp)
    except Exception as e:
        logger.error(
            f"Failed to generate velocity model parameters for realisation {rel_ffp.stem}: {e}"
        )
        rel_ffp.unlink()


if __name__ == "__main__":
    app()
