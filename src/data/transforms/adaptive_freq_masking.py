from typing import List, Union, Tuple
import torch
from torchaudio.transforms import FrequencyMasking


class AdaptiveFrequencyMasking(FrequencyMasking):

    def __init__(self, freq_mask_param: Union[int, Tuple[int, int], List[int]], iid_masks: bool = False):
        """Init function of AdaptiveFreqMasking.

        Args:
            freq_mask (Union[int, Tuple[int, int]]): The frequency mask parameter. If int, it is assumed to be the
                percentage of the frequency dimension to mask. If Tuple[int, int], it is assumed to be the min and max
                number of frequency bins to mask.
            iid_masks (bool, optional): See torchaudio.FrequencyMasking for explanation. Defaults to False.
            p (float, optional): Probability of applying the frequency mask. Defaults to 1.0.
        """
        super().__init__(freq_mask_param=1, iid_masks=iid_masks)
        self.freq_mask_param = freq_mask_param
        if isinstance(freq_mask_param, float):
            assert 0 <= freq_mask_param <= 1.0, "freq_mask_param must be between 0 and 1.0"
        elif isinstance(freq_mask_param, tuple) or isinstance(freq_mask_param, list):
            assert len(freq_mask_param) == 2, "freq_mask_param must be a tuple of length 2"
            assert freq_mask_param[0] >= 0.0, "freq_mask_param[0] must be greater than or equal to 0"
            assert freq_mask_param[1] <= 1.0, "freq_mask_param[1] must be less than or equal to 1.0"
            assert freq_mask_param[0] < freq_mask_param[1],\
                "freq_mask_param[0] must be less than or equal to freq_mask_param[1]"
            if freq_mask_param[0] == 0.0 and freq_mask_param[1] == 1.0:
                raise ValueError("freq_mask_param cannot be the entire frequency dimension")
        else:
            raise ValueError("freq_mask_param must be either an int or a tuple of length 2")
        self.fixed_mask_percentage = isinstance(freq_mask_param, float)

    def forward(self, specgram: torch.Tensor, mask_value=0.0) -> torch.Tensor:
        # We assume that specgram is of shape [..., freq, time]
        freq_size = specgram.size(-2)
        # Now we compute the param for the frequency masking of the superclass. Note: This is not the mask itself,
        # only the size of the mask.
        if self.fixed_mask_percentage:
            freq_mask_param = int(self.freq_mask_param * freq_size)
        else:
            freq_mask_param = torch.randint(
                int(self.freq_mask_param[0] * freq_size),
                int(self.freq_mask_param[1] * freq_size),
                (1,)
            ).item()
        # Here we overwrite the parameter of the superclass before calling its forward method
        self.mask_param = freq_mask_param
        return super().forward(specgram, mask_value=mask_value)