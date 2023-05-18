# Celebrities Face Clustering Service
This repository has the production grade code service built for Face Clustering on cropped Celebrities Face Dataset, obtained from Pinterest.<br>
The dataset is available on Kaggle. Here is the <a href="https://www.kaggle.com/datasets/hereisburak/pins-face-recognition"> Dataset Link !</a>

## Approach
1. Downloading the data from Kaggle, consider whole data or a subset of data for the project, completely your choice. I considered a subset of data due to computing limitations.
2. Building input_data.py for the input processing (creating a sample dataset).
3. Generate 128 dimensional face encodings using face_recognition library, for the cropped face data. Saving the encodings in a pickle file.
4. Create face clusters using the already generated facial embeddings. Create face clusters montages as well. Save the cluster results into unique face folders with a label id.

## Important points to note before Running the Service
1. After cloning the repo, you would need to make changes in the config yaml files. directory/file paths changes are mandatory, wherein you would use your custom paths. Other changes in yaml are optional, you can play with those.
2. You can ignore/delete the input_data.py file, if you are going to consider whole dataset.
3. If you want a subset of data from the whole dataset, you need to make minor changes in the input_data.py file, where I've hardcoded first 100 images from the whole dataset.
4. This service is built considering different environemnts in the mind, like local, dev, staging, prod. If you don't need environment related code, simply remove the code where env_value is mentioned. This is mentioned at a lot of place. Also, a considerable amount of code will be removed from init() function, which is defined at the very starting of each class.
5. The datset is not uploaded in this repository as it's size is huge. You can download the dataset from the above given Kaggle's link.

## Running the Service

### Clone the repo
```
git clone https://github.com/sachelsout/celebrities_face_clustering.git
```

### Install the required libraries
```
pip install -r requirements.txt
```

### Make necessary/optional changes in the config yaml files
Here is one of the yaml files. This is for input. You will need to make changes in the source and destination directories by inserting your custom folders' paths.
Similarly, you would need to make changes in other yaml files as well, where you need to have your custom files/folder paths.
```
local:
  SOURCE_DIR:
    - name: SOURCE_DIR
      value: 'E:/face_clustering/105_classes_pins_dataset'
  DESTINATION_DIR:
    - name: DESTINATION_DIR
      value: 'E:/face_clustering_service/src/DATASETS/sample_dataset'
  NUMBER_OF_IMAGES:
    - name: NUMBER_OF_IMAGES
      value: "10"

dev:
  SOURCE_DIR:
    - name: SOURCE_DIR
      value: 'E:/face_clustering/105_classes_pins_dataset'
  DESTINATION_DIR:
    - name: DESTINATION_DIR
      value: 'E:/face_clustering_service/src/DATASETS/sample_dataset'
  NUMBER_OF_IMAGES:
    - name: NUMBER_OF_IMAGES
      value: "10"
 ```
 
### Run the service
Here while running the app, do mention the env_value as well. env_value is environment value, the environment in which you are going to run the service. e.g. local env, dev env, staging env, prod env.
```
python3 app.py -e <env_value>
```

## Result
Here is the sample montage of clustered faces of a celebrity. It's clearly visible, the service is able to cluster the faces successfully. There are some faces which were unable to cluster, those are stored in a folder with id label as '-1'.

![image](https://github.com/sachelsout/celebrities_face_clustering/assets/86348193/687ab182-55d4-4a27-98b1-a77d8ca43af8)
