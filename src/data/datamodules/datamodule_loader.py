from utils import constants as C

DATA_MODULE_CONSTRUCTORS = {
}


class DataModuleLoader:

    @staticmethod
    def load_data_modules_by_identifiers(config, mean, std):
        for identifier in config[C.KEY_DATASETS]: # -> config["DATASETS"] -> [caers]
            # only searches in the dict keys, not the values
            assert identifier in DATA_MODULE_CONSTRUCTORS, 'The specified datamodule is not known.'
            # TODO We only return one data module here since we do not support joined datasets yet
            return DATA_MODULE_CONSTRUCTORS[identifier](config, mean, std)
