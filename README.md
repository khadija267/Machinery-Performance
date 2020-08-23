# Machinery-Performance:

Deep Learning Model to check whether the Behavior of Machinery is normal or abnormal.

## Machinery:

This model is built and trained on 4 types of machinery(pump,valve,fan, and slide rails) each type has recorded audio of them ( 4 industerial model for each) when they on normal or abnormal behaviour.
link of the dataset :
https://zenodo.org/record/3384388#.X0K7wRmxXjE
link of the published paper:
https://arxiv.org/abs/1909.09347

## Model:

The model is trained as a multi-label classification rather than anomaly detection to make it able to support more class.
Data downloaded is all 4 types in their 6-dB level.
steps of model building: 

### 1- Data Visualizaion:

data visualisation is important in create intuation of the data.
We can find that the 4 machines types is distinguishable from each other and from normal and abnormal cases, but the fan normal and abnormal is tricky.
We can fined that label1 classes are approximately balanced but label2 are not but it's okay as abnormality probability is less than normality.
audio featrues (number of audio channels, sample rate and bit-depth ) summary :
All audios has 1 audio channel(Mono), all have the same sampling rate of 16kHz and that means the comparision will be fair among the audios, and no range of bit depth that means the audios needn't normalisation.

### 2- Data Preparation:

We will extract Mel-Frequency Cepstral Coefficients (MFCC) from the the audio samples. 
MFCC can combine between time domain and frequency domain of an audio accross a sizes window.
To avoid any vary in size after reading the audios via librosa, we make zero-padding and make all arrays size of the audio to be equal.
We made all features arrays flat arrays to be saved in CSV file to train the model into Colab.


