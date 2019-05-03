import logging

class OneLineFormatter(logging.Formatter):
    def formatException(self, exc_info):
        result = super(OneLineFormatter, self).formatException(exc_info)
        return repr(result)

    def format(self, record):
        s = super(OneLineFormatter, self).format(record)
        if record.exc_text:
            s = s.replace('\n', '') + '|'
        return s

def createLogger():
    fh = logging.FileHandler('output.txt', 'w')
    f = OneLineFormatter('%(asctime)s|%(levelname)s|%(message)s|')
    fh.setFormatter(f)
    rootLogger = logging.getLogger()
    rootLogger.addHandler(fh)
    rootLogger.setLevel(logging.DEBUG)
