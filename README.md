# RealTime Facial Expression Recognition in webcam 

### Demo (oh God, gotta have shampoo!)

![gif](https://github.com/Zju-George/realtimeFER/raw/main/assets/livedemo.gif)

### Frameworks

- **Face Dectection** is accomplished by [mediapipe](https://github.com/google/mediapipe) developed by Google.

- **Facial Expression Recognition** is trained by [FER+](https://github.com/microsoft/FERPlus) dataset 
which is held by Microsoft using DCNN (Deep Convolutional Neural Network).

![nn](https://github.com/Zju-George/realtimeFER/raw/main/assets/nnstructure.jpg) 

(Thanks to @[zc](https://github.com/ZC119) of the image)

### Language & Dependencies

- Language: Python3.6
- Dependencies:

    - Pytorch
    - opencv-python
    - mediapipe
    - CUDA10.1 (optional)
    - ...
- you may install all the dependencies via command `python -m pip install -r requirements.txt`
    
### Details
- **Usage**:
```shell script
python camdemo.py --camera 0
```

- **Performance**: 
    - Absolutely **REALTIME**! The model could achieve above the average 60 FPS on a plain PC. If possible, 
    try use a GPU to gain better performance!
    - My poor computer: Intel i7-7700K CPU (4.2GHz) with NVIDIA Quadro P2000 (5G memory)
    
- **Model Structure**: the model is quite simple though. It uses Resnet50 to extract image features and
then it is stacked by two fully connected layer, finally it outputs a 10-size digits vector corresponding to 10 emotion classes.

- **Accuracy**: the model achieves 79.8% accuracy evaluated by FER+ valid subset after 14 epochs of training using softCE loss. 

    | epoch | KLdiv | softCE | weightedSoftCE |
    | :----: | :----: | :----: | :----:|    
    |  0   | 0.005 | 0.005  | 0.005  |
    |  1   | 0.55  | 0.598 | 0.56    |
    |  2   | 0.58  | 0.652 | 0.668   |
    |  3   |   *   | 0.695 | 0.697   |
    |  4   |   *   | 0.726 | 0.71    |
    |  5   |   *   | 0.753 | 0.68    |
    |  6   |   *   | 0.76  | 0.665   |
    | ...  |  ...  | ...   | ...   |
    |  14  |   *   | 0.798 | 0.742 |
  
- **Loss function**: 
    - Rather than original FER, each image in FER+ has been labeled by 10 crowd-sourced 
    taggers but the default implementation of cross-entropy in pytorch uses just one hard label to compute the loss 
    which abandons the information of 10 soft labels. So I implemented the soft cross-entropy to train the model fitting 
    the **probability distribution** of emotion class which got pretty good results.
    
    - One more reason to use softCE loss is that for emotion classification, some human emotions cannot be distinguished 
    well such as happiness and surprise.
    
    - As FER+ is a very **imbalanced** dataset (see image below) so I've tried use weightedSoftCE( like the idea of focal loss)
    but no good which I don't quite get it yet. If you happen to know why, tell me! Also when using weightedSoftCE during training, 
    I found the loss rising upside and down a lot, which means it's not that numerical stable.
    
![data](https://github.com/Zju-George/realtimeFER/raw/main/assets/dataImbalence.png)