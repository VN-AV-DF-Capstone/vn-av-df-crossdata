"""W&B logging (project `vn-av-df-crossdata`, private), on top of vn_av_df.tracking.

Enabled when cross["wandb"] = {"project": ..., "entity": ..., "mode": "online"} (the notebook sets
it from the Kaggle secret WANDB_API_KEY); otherwise every call is a no-op. Only numbers, tables,
charts and result files are logged: never frames or audio of real people.
"""

from pathlib import Path


class Images:
    """Adds image logging to a vn_av_df.tracking.Run."""

    def __init__(self, run):
        self._run = run

    def __getattr__(self, name):
        return getattr(self._run, name)

    def images(self, paths, prefix="figures"):
        if not self._run.active:
            return
        data = {f"{prefix}/{Path(p).stem}": self._run.wandb.Image(str(p)) for p in paths if Path(p).is_file()}
        if data:
            self._run.run.log(data)


def open_run(cross, job_type, name, group=None, config=None, folder=None):
    from vn_av_df.tracking import open_run as base

    options = cross.get("wandb")
    if options:
        options = {"project": cross.get("project", "vn-av-df-crossdata"),
                   "entity": cross.get("entity"), **options}
    cfg = {"wandb": options}
    config = {"seed": cross.get("seed"), **(config or {})}
    return Images(base(cfg, job_type, name, group=group, config=config, folder=folder))
