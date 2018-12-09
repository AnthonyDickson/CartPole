from io import StringIO
import os
from datetime import datetime

import pandas as pd

from utils.path import get_run_path

class Logger:
    """Collects text input for multiple files and writes them into a specific log folder structure.
    
    For example, on the first run of the day the logger may generate a directory like:
        logs/yyyy/mm/dd/run/001
    where yyyy/mm/dd is the date format for year, month, and day.

    If we are logging the files 'a', 'b', and 'c', using the above directory a call to logger.write() would produce 
    the following tree:
    logs/yyyy/mm/dd/run/001
    \t\t\t\t\t        |- a.log
    \t\t\t\t\t        |- b.log
    \t\t\t\t\t        |- c.log
    """
    class Verbosity:
        """Enum capturing the different levels of logger verbosity."""
        SILENT = 0
        MINIMAL = 1
        FULL = 2
        ALL = [SILENT, MINIMAL, FULL]

    def __init__(self, verbosity=Verbosity.SILENT, include_timestamps=True, filename_prefix=''):
        """Make a logger to record text to multiple files.

        Arguments:
            verbosity: the level of verbosity for the logger. See Logger.Verbosity
            include_timestamps: bool flag indicating whether or not to prepend a timestamp to print messages and add a timestamp when writing to file.
            filename_prefix: a string to prefix to filenames.
        """
        log_path = get_run_path(prefix='data/')

        self.verbosity = verbosity
        self.include_timestamps = include_timestamps
        self.filename_prefix = filename_prefix
        self.log_path = log_path
        self.logs = {}

        self.print('Log directory: {}'.format(log_path), Logger.Verbosity.MINIMAL)

    def print(self, msg, min_verbosity=Verbosity.MINIMAL):
        """Print a string message, as long as the logger's verbosity is high enough.

        Arguments:
            msg: the string message to print.
            min_verbosity: the minimum level of verbosity required of the logger for the message to be printed.
        """
        if self.verbosity >= min_verbosity:
            if self.include_timestamps:
                msg = '[{}] {}'.format(datetime.now(), msg)

            print(msg)

    def log(self, filename, contents):
        """Add a entry to the log for a given file.
        
        For example, log('foo', 'bar') adds the contents 'bar' to the log file 'foo'.

        Arguments:
            filename: the file in which the contents should be logged.
            contents: the contents that should be logged.
        """

        try:
            self.logs[filename].append(str(contents))
        except KeyError:
            self.logs[filename] = [str(contents)]

    def clear(self):
        """Clear the log.

        If the log is getting big it is useful to do a write + clear. If you want to write to the same logs later on 
        you can write using the append mode.
        """
        self.logs = {}
        
    def write(self, mode='w', sep='\n'):
        """Write the logs to file.

        Also creates the directory denoted by self.log_path.

        Arguments:
            mode: the file opening mode typical of use with python's open() method.
            sep: the seperator to use to seperate each log entry of a file.
        """
        os.makedirs(self.log_path, exist_ok=True)
        
        self.print('Writing logs to: {}'.format(self.log_path), Logger.Verbosity.MINIMAL)

        for filename in self.logs:
            if len(self.filename_prefix) > 0:
                fullpath = '{}{}.log'.format(self.log_path, self.filename_prefix + '-' + filename)
            else:
                fullpath = '{}{}.log'.format(self.log_path, filename)

            self.print('Writing log file: {}'.format(fullpath), Logger.Verbosity.FULL)

            with open(fullpath, mode) as f:
                contents = sep.join(self.logs[filename])

                if self.include_timestamps:
                    contents = '[{}]\n{}'.format(datetime.now(), contents)

                f.write(contents + '\n')

    def log_to_dataframe(self, name):
        """Convert a log to a pandas DataFrame.

        Strips timestamps.

        Arguments:
            name: the name of the log to convert.

        Returns: the specified log as a DataFrame.
        """
        return pd.read_csv(StringIO('\n'.join(self.logs[name])), comment='[')
