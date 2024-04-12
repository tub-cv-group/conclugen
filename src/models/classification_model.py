from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.cli import instantiate_class
from pytorch_lightning.trainer.connectors.logger_connector.result import _ResultCollection, _ResultMetric
from metrics import ClasswiseWrapper

from models import AbstractModel
from utils import instantiation_util, constants as C


# NOTE: This is a dirty hack! But pytorch lightning doesn't support to return
# metric dictionaries. We inject his method manually to make it work with the
# class-wise wrapper.
def _get_cache(result_metric: _ResultMetric, on_step: bool) -> Optional[torch.Tensor]:
    cache = None
    if on_step and result_metric.meta.on_step:
        cache = result_metric._forward_cache
    elif not on_step and result_metric.meta.on_epoch:
        if result_metric._computed is None:
            # always reduce on epoch end
            should = result_metric.meta.sync.should
            result_metric.meta.sync.should = True
            result_metric.compute()
            result_metric.meta.sync.should = should
        cache = result_metric._computed
    if cache is not None and not result_metric.meta.enable_graph:
        if isinstance(cache, dict):
            return {k: v.detach() for (k, v) in cache.items()}
        else:
            return cache.detach()
    return cache


_ResultCollection._get_cache = staticmethod(_get_cache)


class ClassificationModel(AbstractModel):

    def __init__(
        self,
        num_classes: int,
        losses: Union[dict, List[str]],
        multi_label: bool,
        metric_averaging: str=None,
        labels: List[str]=None,
        dropout_before_classifier: float=None,
        logits_binarization_threshold: float=None,
        recreate_last_layer: bool=True,
        **kwargs
    ):
        """Init function of classification model.
        
        NOTE: The `multi_label` argument will be automatically set from the
        DataModule. But setting `multi_label` to `True` will not change the loss.
        If you want to use a different loss for multi-label classification than
        CrossEntropy you need to pass that in the `loss` argument.

        Args:
            num_classes (int, optional): The number of classes. Defaults to 0.
            recreate_last_layer (bool, optional): Whether to cut off the last
                layer of a pre-trained backbone and re-initialize using the give
                number of classes. Defaults to False.
            metric_balancing (str, optional). The metric averaging
                mode to be used. See Pytorch Lightning docs for help. If set to
                None, no averaging will happen. Defaults to None.
            labels (List[str], optional): The labels of the classes. Used for
                the confusion matrix, for example. Defaults to C.EXPRESSION_LABELS.
            multi_label (bool, optional): Whether this model performs multilabel
                classification. This will not influence the loss (you have to 
                set it yourself to something other than default CrossEntropy) but
                how the predicted classes are computed.
            logits_binarization_threshold (float, optional): Threshold to compute the
                predicted classes in a multi-label classification problem. The
                model can indicate multiple labels as float values and thus a
                binarization threshold is needed to turn the values into 0s
                and 1s. Defaults to None.
            dropout_before_classifier (bool, optional): Whether to add a dropout
                to the output of the last convolutional layer before the linear
                classification layer. Set to the desired dropout rate. Defaults to None.
            loss (dict, optional): The dictionary holding the configuration of
            the loss to use. This dictionary will be passed to Pytorch Lightning's
            instantiate_class function, i.e. you can pass arbitrary class_paths
            and provide any init_args that the respective class supports. The
            same configuration will be used for train, validation and test loss.
            Defaults to CrossEntropyLoss.
        """
        self.recreate_last_layer = recreate_last_layer
        self.num_classes = num_classes
        self.metric_averaging = metric_averaging
        self.labels = labels
        self.multi_label = multi_label
        self.logits_binarization_threshold = logits_binarization_threshold
        if dropout_before_classifier is not None:
            assert 0 <= dropout_before_classifier <= 1, 'The dropout rate must '\
                'be between 0 and 1.'
        self.dropout_before_classifier = dropout_before_classifier
        self.losses = losses

        if logits_binarization_threshold is not None:
            assert multi_label, 'Logits binarization threshold can only be set in multilabel case.'
        if multi_label:
            assert logits_binarization_threshold is not None, 'In multilabel case, you need '\
                'to provide a logits binarization threshold.'

        if num_classes == 0:
            print('Warning: Num classes is 0. Pass it as --model.init_args.num_classes X')

        super().__init__(**kwargs)

    def _load_backbone(self, backbone: str) -> None:
        self._load_backbone_with_feature_dim(backbone, self.num_classes)

    def _load_backbone_with_feature_dim(self, backbone: str, out_feat: int):
        super()._load_backbone(backbone)
        if self.recreate_last_layer:
            num_out_features = self._backbone.classification_layer().out_features
            print('You recreating the last layer of the backbone '
                    f'which had {num_out_features} classes/out features and '
                    f'will now have {out_feat} classes/out features. '
                    'In case you loaded a checkpoint using `load_from_checkpoint`, '
                    'model weights will be properly loaded anyways.')
            last_linear = nn.Linear(
                self._backbone.classification_layer().in_features,
                out_feat)
            nn.init.kaiming_normal_(last_linear.weight)
            if self.dropout_before_classifier is not None:
                self._backbone.set_classification_layer(nn.Sequential(
                    nn.Dropout(self.dropout_before_classifier),
                    last_linear
                ))
            else:
                self._backbone.set_classification_layer(last_linear)
        else:
            print('You chose not to re-create the last layer of the classifier. '
                  'You can trigger such a re-creation using --model.ini_args.recreate_last_layer.')

    def _setup_metrics(self):
        # macro balances the computations based on the number of samples per
        # class, i.e. weights the computed values
        metrics_dict_list = [
            ('train', self._train_metrics),
            ('val', self._val_metrics),
            ('test', self._test_metrics)]
        kwargs = {
            'average': self.metric_averaging
        }
        metric_num = None
        if self.metric_averaging in ['macro', 'weighted']:
            metric_num = self.num_classes
        if self.multi_label:
            kwargs['task'] = 'multilabel'
            kwargs['num_labels'] = metric_num
        else:
            kwargs['task'] = 'multiclass'
            kwargs['num_classes'] = metric_num

        if self.metric_averaging is not None:
            for (subset, metrics_dict) in metrics_dict_list:
                metrics_dict.update({
                    f'{subset}_accuracy':
                        torchmetrics.Accuracy(**kwargs),
                    f'{subset}_precision':
                        torchmetrics.Precision(**kwargs),
                    f'{subset}_recall':
                        torchmetrics.Recall(**kwargs),
                    f'{subset}_f1':
                        torchmetrics.F1Score(**kwargs)
                })
        # If we only have metrics averaging that computes metrics inter-class,
        # we manually also add the accuracy which computes the metrics for each
        # individual class
        if self.metric_averaging in ['macro', 'weighted', 'micro'] or self.metric_averaging is None:
            # We force the average to be None to compute the metrics for each class here.
            kwargs['average'] = None
            for (subset, metrics_dict) in metrics_dict_list:
                metrics_dict.update({
                    f'{subset}_accuracy_class':
                        ClasswiseWrapper(torchmetrics.Accuracy(**kwargs), self.labels),
                    f'{subset}_precision_class':
                        ClasswiseWrapper(torchmetrics.Precision(**kwargs), self.labels),
                    f'{subset}_recall_class':
                        ClasswiseWrapper(torchmetrics.Recall(**kwargs), self.labels),
                    f'{subset}_f1_class':
                        ClasswiseWrapper(torchmetrics.F1Score(**kwargs), self.labels)
                })

    def _setup_losses(self):
        self._train_losses.update({
            # We call it loss not train_loss since PL lightning
            # needs this key in the return dict of the training step
            # for automatic differentiation
            'loss': instantiation_util.instantiate_loss(self.losses),
        })
        self._val_losses.update({
            'val_loss': instantiation_util.instantiate_loss(self.losses),
        })
        self._test_losses.update({
            'test_loss': instantiation_util.instantiate_loss(self.losses),
        })

    def _verify_and_print_class_weights(self, name, weights):
        assert len(weights) ==  self.num_classes, 'Number of class weights must '\
            'match the number of classes.'
        named_weights = dict(zip(self.labels, weights))
        named_weights_str = ', '.join([f'{name}: {weight:.4f}' for name, weight in named_weights.items()])
        print(f'Setting weights on model for {name} classes: {named_weights_str}')

    def _set_new_weight_on_loss(self, loss, weights):
        new_weight = torch.Tensor(weights)
        if isinstance(loss, nn.CrossEntropyLoss):
            loss.register_buffer('weight', new_weight)
        elif isinstance(loss, nn.BCEWithLogitsLoss):
            # BCEWithLogitsLoss also has a weight argument which scales individual
            # batch entires - not what we want here
            loss.register_buffer('pos_weight', new_weight)
        else:
            raise NotImplementedError(f'Loss type {type(loss)} is not yet supported.')

    def set_train_class_weights(self, train_weights: List[float]):
        """Set the class weights to be used in the loss function for an
        imbalanced dataset. Call this function from the DataModule's prepare_data
        function after computing the class weights. 
        
        NOTE: When setting the weights, the loss functions will be re-created if
        the already exist (which they don't if you set balance_classes to True).
        """
        self._verify_and_print_class_weights('train', train_weights)
        self.train_classes_weights = train_weights
        self._set_class_weights_on_train_losses()

    def _set_class_weights_on_train_losses(self):
        for loss in self._train_losses.values():
            self._set_new_weight_on_loss(
                loss, self.train_classes_weights)

    def set_val_class_weights(self, val_weights: List[float]):
        """Set the class weights to be used in the loss function for an
        imbalanced dataset.
        
        NOTE: When setting the weights, the loss functions will be re-created.
        """
        self._verify_and_print_class_weights('val', val_weights)
        self.val_classes_weights = val_weights
        self._set_class_weights_on_val_losses()

    def _set_class_weights_on_val_losses(self):
        for loss in self._val_losses.values():
            self._set_new_weight_on_loss(
                loss, self.val_classes_weights)

    def set_test_class_weights(self, test_weights: List[float]):
        """Set the test class weights to be used in the loss function for an
        imbalanced dataset.
        
        NOTE: When setting the weights, the loss functions will be re-created.
        """
        self._verify_and_print_class_weights('test', test_weights)
        self.test_classes_weights = test_weights
        self._set_class_weights_on_test_losses()

    def _set_class_weights_on_test_losses(self):
        for loss in self._test_losses.values():
            self._set_new_weight_on_loss(
                loss, self.test_classes_weights)

    def forward(self, x: Any):
        """Default classification forward implementation by applying
        softmax to backbone output. Can be modified by subclasses.

        Args:
            x (Any): the batch to process

        Returns:
            torch.Tensor: The classification softmax scores
        """
        logits, features = self._backbone(x)
        results = {
            C.KEY_RESULTS_LOGITS: logits,
            C.KEY_RESULTS_FEATURES: features
        }
        return results

    def extract_targets_for_metric(self, metric, metric_name, targets):
        if self.multi_label:
            return targets.int()
        return targets

    def get_predicted_classes_from_outputs(self, outputs: Any) -> torch.Tensor:
        """Returns the final predicted class from the outputs of the model
        (using e.g. forward). Depending on the implementation of the model,
        this might be the argmax class, etc.

        Args:
            outputs (Any): the outputs produced by the model

        Returns:
            torch.Tensor: the index of the predicted class for each batch
        """
        if isinstance(outputs, dict):
            logits = outputs[C.KEY_RESULTS_LOGITS]
        else:
            logits = outputs
        if self.multi_label:
            sigmoid_logits = torch.sigmoid(logits)
            classes = (sigmoid_logits >= self.logits_binarization_threshold).float()
        else:
            softmax_logits = torch.softmax(logits, dim=1)
            classes = torch.argmax(softmax_logits, dim=1)
        return classes

    def custom_validation_epoch_end_logging(self, outputs):
        if isinstance(outputs, dict):
            self._log_confusion_matrix(
                outputs[C.KEY_RESULTS_LOGITS],
                'Validation Confusion Matrix',
                'val')
        else:
            self._log_confusion_matrix(
                outputs,
                'Validation Confusion Matrix',
                'val')

    def custom_test_epoch_end_logging(self, outputs):
        if isinstance(outputs, dict):
            self._log_confusion_matrix(
                outputs[C.KEY_RESULTS_LOGITS],
                'Test Confusion Matrix',
                'test')
        else:
            self._log_confusion_matrix(
                outputs,
                'Test Confusion Matrix',
                'test')

    def extract_outputs_for_loss(
        self,
        loss,
        loss_name,
        outputs
    ):
        """Extracts the parts of outputs that are relevant for the given loss.
        This method will be overwritten by subclasses that have custom losses, etc.

        Args:
            loss (_type_): the loss to extract the relevant parts for
            outputs (_type_): the outputs to extract parts from

        Returns:
            _type_: the extracted relevant parts of the outputs
        """
        if isinstance(outputs, dict):
            return outputs[C.KEY_RESULTS_LOGITS]
        else:
            return outputs

    def extract_outputs_for_metric(self, metric, metric_name, outputs):
        """Extracts the parts of outputs that are relevant for the given metric.
        This method will be overwritten by subclasses that have custom metrics, etc.

        Args:
            loss (_type_): the metric to extract the relevant parts for
            outputs (_type_): the outputs to extract parts from

        Returns:
            _type_: the extracted relevant parts of the outputs
        """
        if isinstance(outputs, dict):
            return outputs[C.KEY_RESULTS_LOGITS]
        else:
            return outputs
