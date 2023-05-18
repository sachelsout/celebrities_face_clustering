import logging
import os
import yaml
import sys
import face_recognition
import pickle
import cv2


class face_encode_class:

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
                    open(CONFIG_FILE_PATH+'/face_clustering_config.yaml'))
                self.face_detection_method = str(
                    config[env_value]['FACE_DETECTION_METHOD'][0]['value'])
                self.encodings_path = str(
                    config[env_value]['ENCODINGS_PATH'][0]['value'])

            elif env_value in ['stage', 'prod']:
                logging.basicConfig(level=logging.INFO,
                                    format='%(asctime)s.%(msecs)-3d:%(filename)s:%(funcName)s:%(levelname)s:%(lineno)d:%(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S',
                                    stream=sys.stdout, force=True)
                self.face_detection_method = str(os.environ['FACE_DETECTION_METHOD'])
                self.encodings_path = str(os.environ['ENCODINGS_PATH'])

            else:
                raise ValueError

        except Exception as config_ERR:
            logging.error('config_ERR', exc_info=config_ERR)


    def create_face_encodings(self, input_img_paths):
        
        """
        This function creates 128 dimensional facial encodings
        cv2.cvtColor - converts bgr image into rgb image
        To compute bounding boxes and facial encodings, face_recognition library is used, which has associated functions to do the work
        A dictionary created which has image paths, bounding box coordinates and actual facial encodings
        For face_locations(), model used is 'cnn'. You can use 'hog' model as well.
        """
        
        try:

            face_encoding_data = []

            for(i, img_path) in enumerate(input_img_paths):

                logging.info('Processing Image {}/{}'.format(i+1, len(input_img_paths)))
                logging.info(img_path)

                bgr_img = cv2.imread(img_path)
                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                
                bounding_boxes_coords = face_recognition.face_locations(rgb_img, model=self.face_detection_method)
                
                facial_encodings = face_recognition.face_encodings(rgb_img, bounding_boxes_coords)
                
                facial_encodings_listofdicts = [{'image_path': img_path, 'bb_loc': box, 'encodings': enc} for (box, enc) in zip(bounding_boxes_coords, facial_encodings)]
                face_encoding_data.extend(facial_encodings_listofdicts)
            
            return face_encoding_data

        except Exception as dataset_create_ERR:
            logging.error('failed to create sample dataset', exc_info=dataset_create_ERR)
            return False


    def encodings_pickle(self, input_img_paths):
        
        """
        This function dumps the above created facial encodings into the pickle file
        """
        
        try:
            
            logging.info('Dumping the facial encodings obtained, into the pickle file')
            encodings = self.create_face_encodings(input_img_paths)
            
            f = open(self.encodings_path, 'wb')
            f.write(pickle.dumps(encodings))
            f.close()
            
            logging.info('Face encodings saved in {}'.format(self.encodings_path))

        except Exception as save_to_pickle_ERR:
            logging.error('failed to save encodings to a pickle file', exc_info=save_to_pickle_ERR)
            return False
