import os
from pathlib import Path
from dataclasses import dataclass

import ml_tools as mlt  

@dataclass
class RunConfig:

    seed: int

    rel_imdb_ffp: str
    """Relative path to the IMDB file."""

    ignore_events: list[str]
    """List of events to ignore."""

    device: str
    """Device to use"""

    site_inputs: list[str]
    """Model site inputs"""

    source_inputs: list[str]
    """Model source inputs"""

    source_to_site_inputs: list[str]
    """Model source to site inputs"""





    @property
    def imdb_ffp(self) -> Path:
        """
        Get the absolute path to the IMDB file.

        Returns
        -------
        Path
            Absolute path to the IMDB file.
        """
        return Path(os.environ["wdata"]) / self.rel_imdb_ffp

    def to_dict(self) -> dict:
        """
        Convert the RunConfig object to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the RunConfig object.
        """
        return {
            "seed": self.seed,
            "rel_imdb_ffp": self.rel_imdb_ffp,
            "ignore_events": self.ignore_events,
            "device": self.device,
        }
    
    @classmethod
    def from_config_kwargs(cls, config_ffp: Path, **kwargs):
        """
        Creates an instance from the given config.
        If kwargs are set then they overwrite the values
        specified in the config.
        """
        config_dict = mlt.utils.load_yaml(config_ffp)

        for cur_key, cur_val in kwargs.items():
            if cur_val is not None:
                config_dict[cur_key] = cur_val

        return cls(**config_dict)
    

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

    @classmethod
    def from_yaml(cls, ffp: Path):
        return cls.from_dict(mlt.utils.load_yaml(ffp))
