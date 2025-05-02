<h1><p align="center">
Predicting Gaze and Memory patterns of L1 and L2 speakers.
</p></h1>


# Usage

## Visualisations

All visualisations were contained within the [`Data_Exploration.ipynb`](Data_Exploration.ipynb) notebook. To generate these just run the notebook in order and within the blocks labelled `Generate*` change the parameters to switch to different trials.

## Generating the Data sets

- Due to the data being unavailable the generated features had to be created manually and cannot be downloaded from the repo. 
- Within the [`Data_Exploration.ipynb`](Data_Exploration.ipynb) file there is a block labelled `Generate complete data set with derived features`. Run this block to save a complete small data set with derived features to a `.json` file. 
- There is another preprocessing block within the [`deep_learning.ipynb`](deep_learning.ipynb) file for the large data set. Run this file in order until this block is reached. 

### Dataset Augmentation

- For the data set augmentations, uncomment/ modify the last part of the code within the `Generate` block in [`Data_Exploration.ipynb`](Data_Exploration.ipynb) as labelled and change the name of the saved file to something like this: 

```
complete_df.to_json("aug_df.json", orient="records", lines=True)
```

- For the larger data set, uncomment these sections in the `Preprocessing` block as required for the different sequence spreads and Bilingual handling:

#### Bilingual handling:
```
#Uncomment for relevant preprocessing
# Convert BILINGUAL to L1_participant for binary classification
#dataset_df['participent_language'] = 
    dataset_df['participent_language'].apply(
#    lambda x: x[0] if isinstance(x, (np.ndarray, list)) else x
#)
#dataset_df['participent_language'] = 
    dataset_df['participent_language']
        .replace('BILINGUAL', 'L1_participant')

# Drop BILINGUALs
#dataset_df = dataset_df[dataset_df['participent_language'] !=
    'BILINGUAL']
```

#### Spread handling:
```
if features.shape[0] > sequence_length:
    # Middle
    #start = (features.shape[0] - sequence_length) // 2
    #features = features[start:start + sequence_length]
    # End
    #features = features[-sequence_length:]
    # Spread
    indices = np.linspace(0, features.shape[0] - 1,
        sequence_length).astype(int)
    features = features[indices]
```

## Training the Models

- At the bottom of the [`deep_learning.ipynb`](deep_learning.ipynb) file, the `regression` code blocks can be found. If the data has been correctly preprocessed then just running the desired (labelled) models will output their results.
- For the deep learning models, after the `generate` blocks first run the `Preprocessing` block, then the`train/ test split` block in the order presented. Finally either the `transformer` model block should be run or the `LSTM` block. When running the LSTM block comment out or modify the layers for which ever model architecture you wish to use. 

```
    Comment out and uncomment relevant layers:
    
x = Conv1D(64, kernel_size=5, activation='relu')(sequence_input)
x = Dropout(0.2)(x)
x = Conv1D(64, kernel_size=3, activation='relu')(x)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(32))(x)
x = Dropout(0.2)(x)
```
