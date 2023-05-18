import logging
import os
import yaml
import sys
import pickle
import cv2
import numpy as np
from imutils import build_montages
from sklearn.cluster import DBSCAN

class face_cluster_class:

    def __init__(self, env_value=None):
        
        """
        env_value - environment variable value which can be dev or staging or prod (production)
        LOG_FILE_PATH - the path for log file where logs will be stored
        CONFIG_FILE_PATH - the path for configuration files. I've used yaml files to store some configs
        paths set for different operating systems, so that the code can work for different OS.
        NJOBS - how many CPU cores we want to utilize. DBSCAN is a multithreaded algo, so we can use multiple cores of cpu. -1 is default value which means all cpu cores to use
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
                self.encodings_path = str(
                    config[env_value]['ENCODINGS_PATH'][0]['value'])
                self.clustering_result_path = str(
                    config[env_value]['CLUSTERING_RESULT_PATH'][0]['value'])
                self.njobs = int(
                    config[env_value]['NJOBS'][0]['value'])

            elif env_value in ['stage', 'prod']:
                logging.basicConfig(level=logging.INFO,
                                    format='%(asctime)s.%(msecs)-3d:%(filename)s:%(funcName)s:%(levelname)s:%(lineno)d:%(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S',
                                    stream=sys.stdout, force=True)
                self.encodings_path = str(os.environ['ENCODINGS_PATH'])
                self.clustering_result_path = str(os.environ['CLUSTERING_RESULT_PATH'])
                self.njobs = str(os.environ['NJOBS'])

            else:
                raise ValueError

        except Exception as config_ERR:
            logging.error('config_ERR', exc_info=config_ERR)

    
    def load_encodings_pickle(self):
        
        """
        This function loads the pickle file of facial encodings which are obtained
        """
        
        try:
            
            logging.info('Loading face encodings from a pickle file')
            
            encodings_data = pickle.loads(open(self.encodings_path, 'rb').read())
            encodings_data_arr = np.array(encodings_data)
            
            return encodings_data_arr

        except Exception as pickle_load_ERR:
            logging.error('failed to load a pickle file', exc_info=pickle_load_ERR)
            return False

    
    def image_cluster(self, img, id, label_id):
        
        """
        This function creates folders to store respective clusters of images.
        This function will be used later in the next function after clustering is done, to store the clusters of images
        """
        
        try:
            
            cluster_path = self.clustering_result_path + '/label' + str(label_id)
            
            if os.path.exists(cluster_path) == False:
                os.mkdir(cluster_path)
            
            img_file_name = str(id) + '.jpg'
            
            cv2.imwrite(os.path.join(cluster_path, img_file_name), img)

        except Exception as img_cluster_ERR:
            logging.error('failed to put images into the respective clusters', exc_info=img_cluster_ERR)
            return False

    
    def create_face_clusters(self):
        
        """
        This function creates clusters of faces using already generated facial encodings
        DBSCAN - DBSCAN algorithm is used to create face clusterings (DBSCCAN is a density based clustering algorithm)
        DBSCAN trained on facial encodings.
        All the unique faces are clustered in their respective folders
        A separate montage folder is created as well, which will have montage of unique faces each.
        For the images which were unable to cluster into, those are stored in a folder with label -1
        """
        
        try:
            
            dbscan_clt = DBSCAN(metric="euclidean", n_jobs=self.njobs)
            
            encodings_data = self.load_encodings_pickle()
            encodings = [i['encodings'] for i in encodings_data]
            
            dbscan_clt.fit(encodings)
            
            unique_label_ids = np.unique(dbscan_clt.labels_)
            unique_faces_len = len(np.where(unique_label_ids > -1)[0])
            
            logging.info('Total unique faces are {}'.format(unique_faces_len))
            
            os.mkdir(self.clustering_result_path+'/montages')
            
            for label_id in unique_label_ids:
                indexes = np.where(dbscan_clt.labels_ == label_id)[0]
                indexes = np.random.choice(indexes, size=min(25, len(indexes)), replace=False)
                faces_list = []
                
                for i in indexes:
                    img = cv2.imread(encodings_data[i]['image_path'])
                    top, right, bottom, left = encodings_data[i]['bb_loc']
                    face = img[top:bottom, left:right]
                    self.image_cluster(img, i, label_id)
                    face_resize = cv2.resize(face, (96,96))
                    faces_list.append(face_resize)
                
                faces_montage = build_montages(faces_list, (96,96), (5,5))[0]
                
                img_title = "Face ID #{}".format(label_id)
                img_title = "Unknown Faces" if label_id == -1 else img_title
                
                cluster_path = self.clustering_result_path+'/montages'
                cv2.imwrite(os.path.join(cluster_path, img_title+'.jpg'), faces_montage)

        except Exception as create_face_cluster_ERR:
            logging.error('failed to create face clusters', exc_info=create_face_cluster_ERR)
            return False