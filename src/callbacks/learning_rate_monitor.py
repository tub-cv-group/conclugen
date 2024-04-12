from typing import Dict, List
from pytorch_lightning.callbacks import LearningRateMonitor as LRM


class LearningRateMonitor(LRM):

    def __init__(self, **kwargs):
        super(LearningRateMonitor, self).__init__(**kwargs)

    def _add_suffix(self, name: str, param_groups: List[Dict], param_group_index: int, use_names: bool = True) -> str:
        if len(param_groups) > 1:
            if not use_names:
                return f"{name}-pg{param_group_index+1}"
            pg_name = param_groups[param_group_index].get("name", f"pg{param_group_index+1}")
            return f"{name}-{pg_name}"
        elif use_names:
            pg_name = param_groups[param_group_index].get("name")
            return f"{name}-{pg_name}" if pg_name else name
        return name