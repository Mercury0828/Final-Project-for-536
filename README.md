# Final-Project-for-536
Final Project for 536

## Group 3
### group member
    Amey Pimple ap1935@scarletmail.rutgers.edu  
    Jiachen Shen js2884@scarletmail.rutgers.edu  
    Jayashree Domala jd1552@scarletmail.rutgers.edu  
    Shaad Quazi ssq9@scarletmail.rutgers.edu




## Dependency

1. python>=3.10
2. pytorch>=1.11
3. torchvision>=0.12
4. trojanzoo

We use `trojanzoo` in our codes. TrojanZoo is a high-level library for machine learning security. It is designed for attacks and defenses, but could also be used for general machine learning classification tasks.


## Data Pre-process

1. Use provided dataset generation script to generate segmented npz files (with python 2.7).

    > python ./main.py --save-dir ./Dataset/

2. Make a compact version of dataset. Each npz file is composed of 1000 pieces of data.

## Dataset

1. Transform:  
   a. Augment training data by shuffling choices. 

    > Note: This refers to https://github.com/husheng12345/SRAN/blob/master/SRAN/utility/RAVENdataset_utility.py  
    > the shuffling is actually never called in their codes.

3. Dataset Information: 8 classes

## Model
1. Feature Extractor: 4 layer CNN or ResNet18 feature extractor.
    ```python3
        # CNN
        nn.Sequential([
            ConvNormActivation(
                in_channels=16, out_channels=32,
                kernel_size=3, stride=2),
            ConvNormActivation(
                in_channels=32, out_channels=32,
                kernel_size=3, stride=2),
            ConvNormActivation(
                in_channels=32, out_channels=32,
                kernel_size=3, stride=2),
            ConvNormActivation(
                in_channels=32, out_channels=32,
                kernel_size=3, stride=2),
        ])

        # ResNet
        nn.Sequential([conv1, bn1, relu, maxpool,
                       layer1, layer2, layer3, layer4,
                       avgpool])
                       
        # LSTM
        nn.LSTM(input_size=16*4*4, hidden_size=96,
            num_layers=1, batch_first=True)
            
        # WReN
    ```
2. Classifier: 2 layer MLP with dropout.
    ```python3
        nn.Sequential([
            nn.Linear(*, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 8)
        ])
    ```

## Training

1. Optimizer: Adam(lr=kwargs['lr'], betas=(0.9, 0.999), eps=1e-8)
2. lr scheduler: CosineAnnealingLR
3. loss: CrossEntropy
4. Score: top1 accuracy and loss
