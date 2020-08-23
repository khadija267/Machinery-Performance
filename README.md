# Machinery-Performance:

A Keras Deep Learning Model to check whether the Behavior of Machinery is normal or abnormal.

## Machinery:

This model is built and trained on 4 types of machinery(pump,valve,fan, and slide rails) each type has recorded audio of them ( 4 industerial model for each) when They on normal or abnormal behaviour.
link of the dataset :
https://zenodo.org/record/3384388#.X0K7wRmxXjE
link of the published paper:
https://arxiv.org/abs/1909.09347

## Model:

The model is trained as a multi-label classification rather than anomaly detection to make it able to support more class.
Data downloaded is all 4 types in their 6-dB level.
Steps of model building: 

### 1- Data Visualizaion:

Data visualisation is important in create intuation of the data.
We can find that the 4 machines types is distinguishable from each other and from normal and abnormal cases, but the fan normal and abnormal is tricky.
We can fined that label1 classes are approximately balanced but label2 are not but it's okay as abnormality probability is less than normality.
Audio featrues (number of audio channels, sample rate and bit-depth ) summary :
All audios has 1 audio channel(Mono), all have the same sampling rate of 16kHz and that means the comparision will be fair among the audios, and no range of bit depth that means the audios needn't normalisation.

### 2- Data Preparation:

We will extract Mel-Frequency Cepstral Coefficients (MFCC) from the the audio samples. 
MFCC can combine between time domain and frequency domain of an audio accross a sizes window.
To avoid any vary in size after reading the audios via librosa, we make zero-padding and make all arrays size of the audio to be equal.
We made all features arrays flat arrays to be saved in CSV file to train the model into Colab.

### Model Construction:

First we access the data file from GoogleDrive,then we will convert the csv file arrays from string to arrays ( as while saving them in csv format, they are saved as string ), then concatenate the 2 labels we had and to deal with this problem as multi-classification problem.
We econde the categorical labels and split our data.
We construct our CNN of 3 hidden layers, with the help of the Rectifier Linear Unit activation function (relu), and the output layer with 8 neurons ( number of all labels we have) with activation on Softmax.
We compile the model ana then to fit it, and save weights of best of models fitted the data.
Finally we now can calculate model accuracy, and after trying different ones we chose the model of 0.99 percent accuracy for both training and test splits!
As a last step we make prediction via our model, we chose a file from the dataset and fistly it must have padded with the same pad lenght of trained data.
We extract its MFCC by mfcc_feature function and feed the result to our predicction function.




