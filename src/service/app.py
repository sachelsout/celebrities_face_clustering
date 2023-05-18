import os
import sys
import logging
from merger import MergerService
import argparse
import configparser

# OS code (code working on different OS machines)
if os.name == 'nt':
    FULL_PATH = '/'.join((os.path.abspath(__file__).replace('\\',
                         '/')).split('/')[:-1])
elif os.name == 'posix':
    FULL_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])
else:
    raise OSError('untested OS')

# creating log file folder/directory, if it doesn't exists
LOG_FILE_PATH = FULL_PATH + '/logFile'
os.makedirs(LOG_FILE_PATH, exist_ok=True)

# cofig parser for app.py to run on different environments (giving an option for user to specify env_value while running the app)
config = configparser.ConfigParser()
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-e', '--env', help='env specification')
args = vars(parser.parse_args())
env_value = args['env']

if env_value in ['dev', 'local']:
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s.%(msecs)-3d:%(filename)s:%(funcName)s:%(levelname)s:%(lineno)d:%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', 
                        filename=LOG_FILE_PATH +'/logger.log',
                        force=True)
elif env_value in ['stage', 'prod']:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)-3d:%(filename)s:%(funcName)s:%(levelname)s:%(lineno)d:%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', 
                        stream=sys.stdout,
                        force=True)
else:
    raise ValueError

# run the app/service 
if __name__ == "__main__":
    
    try:
        service = MergerService(args['env'])
        logging.info('SERVICE HAS STARTED')
        service.merge()
    
    except Exception as service_ERR:
        logging.error('service error', exc_info=service_ERR)
else:
    logging.critical('service error')