import logging
import logging.handlers


class SingletonType(type):
    def __call__(cls, *args, **kwargs):
        try:
            return cls.__instance
        except AttributeError:
            cls.__instance = super(SingletonType, cls).__call__(*args, **kwargs)
            return cls.__instance


class LogUtil(object):
    __metaclass__ = SingletonType
    _logger = None
    _filename = None

    def __init__(self, filename=None):
        self._filename = filename

        # logger
        self._logger = logging.getLogger('logger')
        # remove default handler
        self._logger.propagate = False

        streamHandler = logging.StreamHandler()
        streamFormatter = logging.Formatter('[%(levelname)8s][%(asctime)s.%(msecs)03d] %(message)s',
                                            datefmt='%Y/%m/%d %H:%M:%S')
        streamHandler.setFormatter(streamFormatter)

        if self._filename is not None:
            file_max_bytes = 10 * 1024 * 1024

            fileHandler = logging.handlers.RotatingFileHandler(filename='./log/' + self._filename,
                                                               maxBytes=file_max_bytes,
                                                               backupCount=10)
            fileFormatter = logging.Formatter('[%(levelname)8s][%(asctime)s.%(msecs)03d] %(message)s',
                                              datefmt='%Y/%m/%d %H:%M:%S')
            fileHandler.setFormatter(fileFormatter)
            self._logger.addHandler(fileHandler)

        self._logger.addHandler(streamHandler)
        self._logger.setLevel(logging.DEBUG)

    def getlogger(self):
        return self._logger
