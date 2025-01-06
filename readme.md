# Music genre predicition system

The scope of this project is to create a deep learning algorithm that aims to as accurately as possible predict the genre of a given music piece
To check later quality of the model, you can do that by running the streamlit app located in `/milPart/app.py`

# Data

Data used to train the model is obtained from the FMA (Free Music Archive) Dataset under the MIT license.
To obtain the data to retrain the models for yourself it is necessary to download it from their github following these steps:
 - download from the FMA GitHub the `fma_large.zip` and `fma_metadata.zip`
 - unzip the folders into the `/data` folder, fma_metadata should be fully unpacked (with parent folder for .csv files being the `/data`)
 - navigate to `/milPart/dataset` and run the `script.py` with arguments of your choice, this should create apropriate folders for training in `/data` folder
 - navigate to `/milPart/model` and run the `main.py` with arguments of your choice, which will result in the training of a given model, located in the `/data/models/[model_name].pth` file.
   - note that running `/milPart/model/main.py` will require you to download additional dependencies listed in `/milPart/model/requirements.txt`
 - additionaly, the training script will provide you with some graphs and statistics referring to the training process of the model

# Requirements

 - To run the program it is first necessary to download all required dependencies located in `requirements.txt`
 - If all dependencies are installed, you can run the app using command `streamlit run app.py`
 - Then if everything was set up correctly, you should see a model selector, upon selecting which a file upload will pop up, where you can upload your piece of music. Upon uploading you will be displayed a spectrogram, along with a "predict" button which will feed the image into the model, and give you back its prediciton.


