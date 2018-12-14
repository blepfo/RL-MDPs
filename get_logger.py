import logging

from typing import Optional


DEFAULT_LOGGER_LEVEL=logging.INFO

def get_logger(output_file: Optional[str]=None,
               logger_level: int=DEFAULT_LOGGER_LEVEL,
               file_level: Optional[int]=None,
               stream_level: Optional[int]=None,
               output_dir: str=None):
    """Get logger for the genenet package.
    Returns a logger for the genenet package with two handlers:
    a file handler to write logs to an output file, and
    a stream handler to write logs to the console.
    Args:
        output_file (str): File where the logger should save to.
        logger_level (Optional[int]): Minimum level of messages to be logged.
        file_level (Optional[int]): Minimum level of messages to be saved to
            file.
        stream_level (Optional[int]): Minimum level of messages to be printed
            to console.
        output_dir (str): Directory for output file to be saved in. Defaults
            to current directory.
    Returns:
        (logging.logger): Logger for the genenet package.
    """
    new_logger = logging.getLogger('lmdp')
    new_logger.setLevel(logger_level)
    # Use the same level as logger_level for everything else by default
    if file_level is None:
        file_level = logger_level
    if stream_level is None:
        stream_level = logger_level

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S')

    if output_file is not None:
        # Log to output to file
        if output_dir is not None:
            # Save logfile to output_dir
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            output_file = os.path.join(output_dir, output_file)

        filehandler = logging.FileHandler(output_file)
        filehandler.setLevel(file_level)
        filehandler.setFormatter(formatter)
        new_logger.addHandler(filehandler)

        # Log to console
        streamhandler = logging.StreamHandler()
        streamhandler.setLevel(stream_level)
        streamhandler.setFormatter(formatter)
        new_logger.addHandler(streamhandler)

        new_logger.debug('Created logger with log file {}'.format(
            output_file))

    return new_logger
