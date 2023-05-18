import logging
import os
import shutil
import yaml
import sys
from imutils import paths

class input_class:

    def __init__(self, env_value=None):

        """
        env_value - environment variable value which can be dev or staging or prod (production)
        LOG_FILE_PATH - the path for log file where logs will be stored
        CONFIG_FILE_PATH - the path for configuration files. I've used yaml files to store some configs
        paths set for different operating systems, so that the code can work for different OS.
        """

        try:

            if os.name == 'nt':
                LOG_FILE_PATH = '/'.join(
                    (os.path.abspath(__file__).replace('\\', '/')).split('/')[:-3]) + '/logFile'
                CONFIG_FILE_PATH = '/'.join(
                    (os.path.abspath(__file__).replace('\\', '/')).split('/')[:-3]) + '/config'
                os.makedirs(LOG_FILE_PATH, exist_ok=True)

            elif os.name == 'posix':
                LOG_FILE_PATH = '/'.join(os.path.abspath(
                    __file__).split('/')[:-1]) + '/logFile'
                CONFIG_FILE_PATH = '/'.join(os.path.abspath(
                    __file__).split('/')[:-3]) + '/config'
                os.makedirs(LOG_FILE_PATH, exist_ok=True)
            else:
                raise OSError('untested OS')

        except Exception as file_dir_ERR:
            logging.error('file_dir_ERR', exc_info=file_dir_ERR)

        try:

            if env_value == None:
                env_value = 'dev'

            if env_value in ['local', 'dev']:
                logging.basicConfig(level=logging.INFO,
                                    format='%(asctime)s.%(msecs)-3d:%(filename)s:%(funcName)s:%(levelname)s:%(lineno)d:%(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S',
                                    filename=LOG_FILE_PATH + '/logger.log',
                                    force=True)
                config = yaml.safe_load(
                    open(CONFIG_FILE_PATH+'/input_config.yaml'))
                self.source_dir = str(
                    config[env_value]['SOURCE_DIR'][0]['value'])
                self.destination_dir = str(
                    config[env_value]['DESTINATION_DIR'][0]['value'])
                self.images_len = int(
                    config[env_value]['NUMBER_OF_IMAGES'][0]['value'])

            elif env_value in ['stage', 'prod']:
                logging.basicConfig(level=logging.INFO,
                                    format='%(asctime)s.%(msecs)-3d:%(filename)s:%(funcName)s:%(levelname)s:%(lineno)d:%(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S',
                                    stream=sys.stdout, force=True)
                self.source_dir = str(os.environ['SOURCE_DIR'])
                self.destination_dir = str(os.environ['DESTINATION_DIR'])
                self.images_len = int(os.environ['NUMBER_OF_IMAGES'])

            else:
                raise ValueError

        except Exception as config_ERR:
            logging.error('config_ERR', exc_info=config_ERR)


    def create_sample_dataset(self):
        
        """
        This function takes actual 105 classes pins dataset and returns a sample dataset out of it. I am not considering whole dataset here
        Considering only first 100 images from the dataset
        """
        
        try:
            src_dir_path = self.source_dir
            dest_dir_path = self.destination_dir
            number_of_images = self.images_len

            if os.path.exists(self.destination_dir) == False:
                logging.info('Creating a directory for sample dataset')
                os.mkdir(dest_dir_path)

            logging.info('creating a sample dataset from a large dataset')
            for path, subdirs, files in os.walk(src_dir_path):
                for name in files[:number_of_images]:
                    filename = os.path.join(path, name)
                    shutil.copy2(filename, dest_dir_path)
            
            logging.info('selecting first 100 images from the sampled dataset')
            image_paths = list(paths.list_images(dest_dir_path))
            first_100_image_paths = image_paths[:100]
            return first_100_image_paths

        except Exception as dataset_create_ERR:
            logging.error('failed to create sample dataset', exc_info=dataset_create_ERR)
            return False
