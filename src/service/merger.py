import logging
import os
import sys

# importing other required .py files/classes
from input.v1.input_data import input_class
from encoding.v1.face_encoding import face_encode_class
from clustering.v1.face_clustering import face_cluster_class

class MergerService:

    def __init__(self, env_value=None):
        
        """
        env_value - environment variable value which can be dev or staging or prod (production)
        LOG_FILE_PATH - the path for log file where logs will be stored
        paths set for different operating systems, so that the code can work for different OS.
        Required classes called and self variable assigned to them
        """
        
        try:
            
            if os.name == 'nt':
                LOG_FILE_PATH = '/'.join(
                    (os.path.abspath(__file__).replace('\\', '/')).split('/')[:-1]) + '/logFile'
                os.makedirs(LOG_FILE_PATH, exist_ok=True)

            elif os.name == 'posix':
                LOG_FILE_PATH = '/'.join(os.path.abspath(
                    __file__).split('/')[:-1]) + '/logFile'
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

            elif env_value in ['stage', 'prod']:
                logging.basicConfig(level=logging.INFO,
                                    format='%(asctime)s.%(msecs)-3d:%(filename)s:%(funcName)s:%(levelname)s:%(lineno)d:%(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S',
                                    stream=sys.stdout, force=True)

            else:
                raise ValueError

            self.input = input_class(env_value=env_value)
            self.face_encode = face_encode_class(env_value=env_value)
            self.face_cluster = face_cluster_class(env_value=env_value)

        except Exception as config_ERR:
            logging.error('config_ERR', exc_info=config_ERR)

    
    def merge(self):
        
        """
        This function basically executes the code from different .py files in a way we want. 
        e.g. here we consider input and its processing first, then passing that input to create facial embeddings, then at last creating clusters
        """
        
        try:
            
            sample_dataset = self.input.create_sample_dataset()
            self.face_encode.encodings_pickle(input_img_paths=sample_dataset)
            self.face_cluster.create_face_clusters()

        except Exception as merge_ERR:
            logging.error('failed to merge the classes and the functions', exc_info=merge_ERR)
            return False
