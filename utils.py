from datetime import datetime
import os

class Utils:
    @staticmethod
    def get_run_path(prefix=''):
        if prefix[-1] != '/':
            prefix = prefix + '/'    
    
        now = datetime.now()

        path = prefix
        path += '/'.join(map(lambda x: '{:02d}'.format(x), [now.year, now.month, now.day]))
        path += '/run'

        if not os.path.isdir(path):
            path += '/001'
        else:
            run_number = 1

            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_dir() and entry.name == '{:03d}'.format(run_number):
                        run_number += 1

            path += '/{:03d}'.format(run_number)

        return path + '/'