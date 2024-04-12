from typing import List, Union, Tuple
import torch
from torchaudio.transforms import TimeMasking


class AdaptiveTimeMasking(TimeMasking):

    def __init__(self, time_mask_param: Union[int, Tuple[int, int], List[int]], iid_masks: bool = False):
        """Init function of AdaptiveTimeMasking.

        Args:
            freq_mask (Union[int, Tuple[int, int]]): The frequency mask parameter. If int, it is assumed to be the
                percentage of the frequency dimension to mask. If Tuple[int, int], it is assumed to be the min and max
                number of frequency bins to mask.
            iid_masks (bool, optional): See torchaudio.FrequencyMasking for explanation. Defaults to False.
            p (float, optional): Probability of applying the frequency mask. Defaults to 1.0.
        """
        super().__init__(time_mask_param=1, iid_masks=iid_masks)
        self.time_mask_param = time_mask_param
        if isinstance(time_mask_param, float):
            assert 0 <= time_mask_param <= 1.0, "time_mask_param must be between 0 and 1.0"
        elif isinstance(time_mask_param, tuple) or isinstance(time_mask_param, list):
            assert len(time_mask_param) == 2, "time_mask_param must be a tuple of length 2"
            assert time_mask_param[0] >= 0.0, "time_mask_param[0] must be greater than or equal to 0"
            assert time_mask_param[1] <= 1.0, "time_mask_param[1] must be less than or equal to 1.0"
            assert time_mask_param[0] < time_mask_param[1],\
                "time_mask_param[0] must be less than or equal to time_mask_param[1]"
            if time_mask_param[0] == 0.0 and time_mask_param[1] == 1.0:
                raise ValueError("time_mask_param cannot be the entire frequency dimension")
        else:
            raise ValueError("time_mask_param must be either an int or a tuple of length 2")
        self.fixed_mask_percentage = isinstance(time_mask_param, float)

    def forward(self, specgram: torch.Tensor, mask_value=0.0) -> torch.Tensor:
        # We assume that specgram is of shape [..., freq, time]
        time_size = specgram.size(-1)
        # Now we compute the param for the time masking of the superclass. Note: This is not the mask itself,
        # only the size of the mask.
        if self.fixed_mask_percentage:
            time_mask_param = int(self.time_mask_param * time_size)
        else:
            time_mask_param = torch.randint(
                int(self.time_mask_param[0] * time_size),
                int(self.time_mask_param[1] * time_size),
                (1,)
            ).item()
        # Here we overwrite the parameter of the superclass before calling its forward method
        self.mask_param = time_mask_param
        return super().forward(specgram, mask_value=mask_value)