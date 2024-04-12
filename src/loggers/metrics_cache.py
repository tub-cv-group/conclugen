import logging
from typing import List
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from collections import OrderedDict


logger = logging.getLogger(__name__)


class MetricsCacheLogger(Logger):
    """A logger that caches metrics in memory to allow accessing them."""

    def __init__(self, metrics_to_cache: List[str], suppress_warnings: bool = True):
        super().__init__()

        self.cached_metrics = {}
        self.cached_stepped_metrics = {}
        self.metrics_to_cache = metrics_to_cache
        self.logger_name = 'metrics_cache'
        self.suppress_warnings = suppress_warnings

    @property
    def name(self):
        return "Metrics cache logger"

    @property
    def version(self):
        return "1.0"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    def reset(self):
        self.cached_metrics = {}
        self.cached_stepped_metrics = {}

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        for metric_name in self.metrics_to_cache:
            if metric_name in metrics:
                if metric_name not in self.cached_metrics:
                    self.cached_metrics[metric_name] = []
                self.cached_metrics[metric_name].append(metrics[metric_name])
                if step is not None:
                    if metric_name not in self.cached_stepped_metrics:
                        # To get the steps in order
                        self.cached_stepped_metrics[metric_name] = OrderedDict()
                    self.cached_stepped_metrics[metric_name][step] = metrics[metric_name]
            elif not self.suppress_warnings:
                logger.warning(f"Metric {metric_name} not found in metrics dictionary with keys: {metrics.keys()}.")

    def log_hyperparams(self, params):
        # Nothing to do here.
        pass

    def log_artifact(self, path):
        # Nothing to do here.
        pass

    def log_image(self, **kwargs):
        pass

    def log_code(self, source_dir: str):
        pass

    def log_model(self, file_path: str, file_name_to_log: str):
        pass

    def log_html(self, filename: str, html):
        pass

    def log_table(self, filename: str, tabular_data):
        pass

    def log_histogram_3d(self, values, name=None, epoch=None, step=None):
        pass

    def log_figure(
        self, figure_name: str, figure,
        prefixes: List[str] = None, overwrite: bool = False, step: int = None
    ):
        pass

    def log_confusion_matrix(self, preds, targets, name, epoch, labels, file_prefix):
        pass

        