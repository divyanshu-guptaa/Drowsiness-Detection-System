
<h1> Drowsiness-Detection-System </h1>
Drowsiness Detection System is made using OpenCV, TensorFlow, pre-trained model InceptionV3 and mrl eye dataset.

<h2>Step 1 Installing necessary libraries </h2>

- Tensorflow (Preferred version 2.13.0)

```
pip install tensorflow
```

- OpenCV (Preferred version 4.8.0)

```
pip install opencv-python
```

- NumPy (Preferred version 1.24.3)

```
pip install numpy
```

- Sklearn (Preferred version 1.2.1)

```
pip install -U scikit-learn
```

- Pygame (Preferred version 2.5.2)

```
pip install pygame
```

<h2>Step 2 Downloading the dataset</h2>
The MRL Eye Dataset is a large collection of infrared images of human eyes.

The pre-processed dataset in .zip format can be downloaded from [here](https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset)

<br/>

>[!NOTE]
>Move folders named train and test under a new folder named input inside current workspace

<br/>

<h2>Step 3 Running files</h2>

First run `new_model.py`. A new file named `my_new_model.keras` will be created. Now run `drowsiness detection.py` after allowing camera access.
