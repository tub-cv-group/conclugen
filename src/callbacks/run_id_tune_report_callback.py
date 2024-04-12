from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning import Callback, Trainer, LightningModule


class RunIDTuneRportCallback(TuneReportCallback):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.run_id = None

    def _get_report_dict(self, trainer: Trainer, pl_module: LightningModule):
        assert self.run_id is not None, 'run_id is None but should be set'
        if trainer.sanity_checking:
            return
        report_dict = super()._get_report_dict(trainer, pl_module)
        report_dict['run_id'] = self.run_id
        return report_dict
