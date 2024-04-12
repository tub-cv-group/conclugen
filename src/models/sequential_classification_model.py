from typing import Any
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss, CrossEntropyLoss, NLLLoss

from models import ClassificationModel
from utils import constants


class SequentialClassificationModel(ClassificationModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # NOTE we also support BCELoss and NLLLoss here since the user might
        # prefer to not make use of this trick we implemented to sneack the
        # averaged sigmoids or averaged softmaxes to BCEWithLogitsLoss or
        # CrossEntropyLoss

    def forward(self, x: Any):
        if not isinstance(x, list):
            # x is not a list of chunks of the sequence samples e.g. during training
            # when random chunks are picked and returned
            return self._backbone(x)
        else:
            # This is the case for validation and testing, we land here since
            # all chunks of a sample have been loaded and returned
            outputs = []
            for inputs in x:
                # Process all chunks of the inputs
                outputs.append(self._backbone(inputs))
            return outputs

    def _avg_sigmoids(self, outputs):
        outputs = torch.sigmoid(outputs)
        # Average the probabilities of the individual sequence chunks
        outputs = outputs.mean(dim=1)
        return outputs

    def extract_outputs_for_loss(self, loss, loss_name, outputs):
        if isinstance(outputs, list):
            # We are in the case where the outputs is a list, i.e. multiple chunks
            # of a sequence
            logits = outputs[constants.KEY_RESULTS_LOGITS]
            logits = torch.stack(logits, dim=1)
            if isinstance(loss, NLLLoss):
                logits = torch.softmax(logits, dim=2)
                logits = logits.mean(dim=1)
                return logits
            elif isinstance(loss, CrossEntropyLoss):
                # Same here as for the BCEWithLogitsLoss, we compute the inverse of
                # the softmax to sneak through our averaged softmax
                softmax_logits = torch.softmax(logits, dim=2)
                avg_softmax_logits = softmax_logits.mean(dim=1)
                C = torch.sum(torch.exp(avg_softmax_logits), dim=1)
                # + C is actually not needed but we'll put it here anyways
                inverse_softmax = torch.log(avg_softmax_logits) + C
                return inverse_softmax
            elif isinstance(loss, BCEWithLogitsLoss):
                # This loss applies a sigmoid layer internally so we cannot apply
                # a sigmoid layer directly. We apply a trick and compute the
                # average sigmoid values and then compute the inverse of the sigmoid
                # function. Applying the sigmoid in the loss will then again yield
                # the averaged sigmoids.
                # See https://stats.stackexchange.com/questions/581766/how-to-calculate-sigma-1-frac1n-sum-i-1n-sigmax-i-in-a-numer
                avg_sigmoids = self._avg_sigmoids(logits)
                # This is the inverse of the sigmoid, meaning if the BCEWithLogitsLoss
                # computes the sigmoid of it, we'll end up with the average sigmoids
                # computed above
                inverse_sigmoid = torch.log(0.0001 + avg_sigmoids / (0.0001 + 1 - avg_sigmoids))
                return inverse_sigmoid
            elif isinstance(loss, BCELoss):
                # For BCELoss, which doesn't have a sigmoid layer like BCEWithLogitsLoss,
                # we can simply compute the sigmoid and then take the mean
                logits = self._avg_sigmoids(logits)
                return logits
            else:
                raise NotImplementedError(f'Loss type {type(loss)} needs to be implemented first.')
        else:
            if isinstance(loss, NLLLoss):
                outputs = torch.softmax(outputs[constants.KEY_RESULTS_LOGITS], dim=1)
            elif isinstance(loss, BCELoss):
                outputs = self._avg_sigmoids(outputs[constants.KEY_RESULTS_LOGITS])
            else:
                outputs=outputs[constants.KEY_RESULTS_LOGITS]
            return outputs

    def extract_outputs_for_metric(self, metric, metric_name, outputs):
        if isinstance(outputs, list):
            outputs = torch.stack(outputs, dim=1)
            if self.multi_label:
                avg_sigmoids = self._avg_sigmoids(outputs)
                return avg_sigmoids
            else:
                softmax_logits = torch.softmax(outputs, dim=2)
                # Averae the probabilities of the individual sequence chunks
                avg_softmax_logits = softmax_logits.mean(dim=1)
                return avg_softmax_logits
        else:
            return super().extract_outputs_for_metric(metric, metric_name, outputs)

    def get_predicted_classes_from_outputs(self, outputs: Any) -> torch.Tensor:
        if isinstance(outputs, list):
            outputs = torch.stack(outputs, dim=1)
            if self.multi_label:
                sigmoid_logits = self._avg_sigmoids(outputs)
                classes = (sigmoid_logits >= self.logits_binarization_threshold).float()
                classes = (classes < self.logits_binarization_threshold).float()
            else:
                softmax_logits = torch.softmax(outputs, dim=2)
                # Averae the probabilities of the individual sequence chunks
                avg_softmax_logits = softmax_logits.mean(dim=1)
                classes = torch.argmax(avg_softmax_logits, dim=1)
            return classes
        else:
            return super().get_predicted_classes_from_outputs(outputs)