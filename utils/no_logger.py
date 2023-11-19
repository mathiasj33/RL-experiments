from utils.logger import Logger, LogMetadata


class NoLogger(Logger):
    """
    A dummy logger that does not perform any logging. Useful for debugging without creating logs.
    """

    def __init__(self, metadata: LogMetadata, keys: set[str], stds: set[str]):
        super().__init__(metadata, keys, stds)

    def log_metadata(self):
        pass

    def log(self):
        pass

    def save_model(self, model):
        pass

    def finish(self):
        pass
