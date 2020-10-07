# Machinery-Performance:

A Keras Deep Learning Model to check whether the Behavior of Machinery is normal or abnormal.

## Machinery:

This model is built and trained on 4 types of machinery(pump, valve, fan, and slide rails) each type has recorded audio of them ( 4 industrial models for each) when They on normal or abnormal behaviour. link of the dataset: https://zenodo.org/record/3384388#.X0K7wRmxXjE link of the published paper: https://arxiv.org/abs/1909.09347

## Model:

The model is trained as a multi-label classification rather than anomaly detection to make it able to support more class. Data downloaded are all 4 types in their 6-dB level. Steps of model building:

# 1- Data Visualization:

Data visualisation is important in create intuition of the data. We can find that the 4 machines types are distinguishable from each other and normal and abnormal cases, but the fan normal and abnormal is tricky. We can find that label1 classes are approximately balanced but label2 is not but it's okay as abnormality probability is less than normality. Audio features (number of audio channels, sample rate and bit-depth ) summary: All audios have 1 audio channel(Mono), all have the same sampling rate of 16kHz and that means the comparison will be fair among the audios, and no range of bit depth that means the audios needn't normalisation.

# 2- Data Preparation:

We will extract Mel-Frequency Cepstral Coefficients (MFCC) from the audio samples. MFCC can combine between time domain and frequency domain of audio across a sizes window. To avoid any variation in size after reading the audios via Libros, we make zero-padding and make all arrays size of the audio to be equal. We made all features arrays flat arrays to be saved in CSV file to train the model into Colab.

# 3- Model Construction:

First, we access the data file from GoogleDrive, then we will convert the CSV file arrays from string to arrays ( as while saving them in CSV format, they are saved as string ), then concatenate the 2 labels we had and deal with this problem as a multi-classification problem. We encode the categorical labels and split our data. We construct our CNN of 3 hidden layers, with the help of the Rectifier Linear Unit activation function (relu), and the output layer with 8 neurons ( number of all labels we have) with activation on Softmax. We compile the model and then to fit it, and save weights of best of models fitted the data. Finally, we now can calculate model accuracy, and after trying different ones we chose the model of 0.99 per cent accuracy for both training and test splits! As the last step, we predict our model, we chose a file from the dataset and firstly it must have padded with the same pad length of trained data. We extract its MFCC by mfcc_feature function and feed the result to our prediction function.

Packages Used :

Tensorflow = 2.2 -- Keras -- Librosa -- Pandas -- Numpy -- Scikit-learn -- Matplotlib.
