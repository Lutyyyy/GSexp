import argparse 
import logging
import os
import time
from datetime import timedelta
from .util_print import Bcolors 


class LogFormatter:
    COLORS = {
        "DEBUG": Bcolors.DEBUG,
        "INFO": Bcolors.DARKCYAN,
        "WARNING":Bcolors.WARNING,
        "ERROR": Bcolors.FAIL,
        "CRITICAL": Bcolors.HEADER,
        "ENDC": Bcolors.ENDC,
    }

    def __init__(self, use_color=False):
        self.start_time = time.time()
        self.use_color = use_color

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        color = self.COLORS.get(record.levelname, "") if self.use_color else ""
        endc = self.COLORS["ENDC"] if self.use_color else ""

        prefix = "%s%s - %s - %s%s" % (
            color,
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
            endc,
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def create_logger(logpath, rank=0):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    file_log_formatter = LogFormatter(use_color=False)
    console_log_formatter = LogFormatter(use_color=True)

    # create file handler and set level to debug
    if logpath is not None:
        filepath = "%s-%i" % (os.path.join(logpath, 'train.log'), rank) if rank > 0 else os.path.join(logpath, 'train.log')
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logpath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        file_log_formatter.start_time = time.time()
        console_log_formatter.start_time = time.time()
    
    # set log command
    def log_command(args):
        return _log_command(logger, os.path.join(logpath, 'cmd.txt'), args)
    
    logger.log_command = log_command
    logger.reset_time = reset_time

    return logger


def _log_command(logger, filepath, args):
    if isinstance(args, argparse.Namespace):
        command_str = "Command: %s" % " ".join(f"--{k} {v}" for k, v in vars(args).items())
    else:
        command_str = "Command: %s" % " ".join(args) if args else "No command arguments provided."
    logger.warning(command_str)
    if filepath is not None:
        with open(filepath, "a") as f:
            f.write(command_str + '\n')


# _logger_instance = None

# def get_logger(filepath=None, rank=0):
#     global _logger_instance
#     if _logger_instance is None:
#         _logger_instance = create_logger(filepath, rank)
#     return _logger_instance
