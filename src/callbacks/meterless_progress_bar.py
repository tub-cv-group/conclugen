from pytorch_lightning.callbacks import TQDMProgressBar

class MeterlessProgressBar(TQDMProgressBar):

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar