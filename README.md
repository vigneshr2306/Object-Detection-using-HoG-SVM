# Object Detection using Histogram of Gradients and Support Vector Machine

## Code Executing Steps

1. `python3 train_phone_finder.py`
2. `python3 test_phone_finder.py -i /path/to/test_image`

# Pipeline

1. The dataset is initally preprocessed using the `preprocess.py` to create the positive and negative features used by the Histogram of Gradients. Execute it by using `python preprocess.py`. This will create the train_pos, train_neg, test_pos, test_neg in the dataset folder.
2. The `extract-features.py` is used to generate the Histogram of Gradient feature vectors. Execute it by using `python extract-features.py`.
3. The `train_phone_finder.py` is used to train a model using Support Vector Machine with the generated feature vector data. Execute it by using `python train_phone_finder.py`
4. The `test_phone_finder.py` is used to test the SVM model and output the centroid of the phone. This involves sliding window of 50x50 and stride of 2x2 to iterate through the image and find the location of the phone. Non-Maximum Suppression is used to reduce the number of bounding boxes based on a threshold. Based on the maximum confidence score, the real location of the box is found. Centroid is calculated from the bounding box coordinates. The path of the image is given as argument and can be executed as `python3 test_phone_finder.py -i /path/to/test_image`.
5. The ``test-classifier.py` is used to iterate through the entire test set (20 images) and calculates Mean Square Error with respect to actual and calculated centroid. The mean of all the error is used to calculate the accuracy of the model, which turned out to be
   99.08%
