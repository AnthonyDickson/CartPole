import os
import re
from datetime import datetime

def get_run_path(prefix=''):
        """Generate a path string in the format 'prefix/yyyy/mm/dd/run/nnn/'.

        Arguments:
            prefix: the path to prepend to the run path.

        Returns: a path string in the format 'prefix/yyyy/mm/dd/run/nnn/' 
                 where prefix is the parameter 'prefix', yyyy/mm/dd is the date, and nnn is the 3-digit zero-padded run number.
        """
        if len(prefix) > 0 and prefix[-1] != '/':
            prefix = prefix + '/'    
    
        now = datetime.now()

        path = prefix
        path += '/'.join(map(lambda x: '{:02d}'.format(x), [now.year, now.month, now.day]))
        path += '/run'

        if not os.path.isdir(path):
            path += '/001'
        else:
            run_number = 1
            pattern = re.compile("[0-9]{3}")

            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_dir() and re.match(pattern, entry.name):
                        run_number = max(run_number, int(entry.name) + 1)

            path += '/{:03d}'.format(run_number)

        return path + '/'
