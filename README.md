# NEURAL NETWORK , IMAGE RECOGNITION AND IMAGE OPERATIONS

## Andrei Muravev
###     9/27/2021


## Introduction:

 As one of the prominent directions in the computer vision and AI, the *facial emotion recognition*  or FER plays an important role in people's daily work and life. Many companies including NEC Corporation from Japan  or Google invest in FER’s technologies which proves its growing acceptance. FER is based on learning and analyzing different positions of nose, eyebrows or lips  from an image or video to determine the psychological emotions of the object. Facial emotions can be divided into seven universal categories: ***happy, fearful, angry, surprised, disgusted, sad and neutral***.
  
   Potential ***uses of FER*** could be in:
  * sport to monitor physical conditions, in personal training app's, Fitbit.
  * analyze facial expressions to predict reaction to movies or music, for example,  to make recommendations.
  * analyze customer's emotions and satisfactions with a store or products
  * detect diseases in healthcare (depressions, disorders, suicide) or even observe patients in hospitals
  * monitor students or employees attention and involvements
  * public safety and security, but need to consider other factors as well. For example, a person missing a flight would act suspicious too.
  * ***driving systems and fatigue detection!!!***
  
  ***Difficulties:***
  * Data could be not accurate since emotions could vary from person to person
  * Some people could mix emotions or may not express emotions at all
  * Particular attention needs to be provided to determination of special states of faces such as sarcasm
  * Need to consider lightning factors and covered faces with masks, for example
  * Consider age, sex and race 


## Data sources:

   All datasets and images could be dowbloaded from these two links below. I didn't include them to my repo because they are too large.

- [Art collection from Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

- [Kaggle's Dataset of Facial Expression](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)



## Problem Statement:

 In my work, I've included four different notebooks and each one of them has different tasks:
 
 *  ***Emotions Recognition and Neural Network*** notebook. Teach the machine to recognize emotions.
 *  ***Resnet50---VGG16*** notebook. Implement two models for very deep neural network.
 *  ***Art Work*** notebook. Explore and experiment with different Image Operations using ***PIL***, ***OpenCV*** and ***Matplotlib*** libraries.
 *  ***Neural Style Transfer*** notebook. Combine two images together to get a new image using a technique called 'neural style transfer.'





## Notebook's Content:

### 1. Art Work Notebook

1. ***Glob*** and ***Path*** to Import and Load Images.
2. ***Python Image Library***
3. ***OpenCV*** and functions:cv.imread(),cv.getRotationMatrix2D(),cv.cvtColor(),cv.threshold(),cv.GaussianBlur(), cv.putText().
4. ***CascadeClassifier*** and detecting faces.
5. ***Matplotlib*** and Image operations.
6. ***ImageDataGenerator()***

### 2. Emotions Recognition and Neural Network Notebook

1. ***Importing*** Images using best practices from the 'Art Work' notebook
2. ***Preprocessing*** 
3. ***Instantiating*** the ConvNet
4. ***Overfitting***. Dropouts, BatchNormalization, CallBacks( EarlyStopping, ModelCheckpoint, ReduceLROnPlateau )
5  ***Augmentation***
6. ***Vizualization***
7. ***Classification Report*** and ***Confusion Matrix***
8. ***Image Predictions*** 
9. ***Conclusion and observations***


### 3. Resnet---VGG16 Notebook'

The main purpose of this notebook is to use two pre-trained models(***Resnet50 and VGG16***) to predict emotions from my second notebook. Here I've included a small ***introduction*** to those models as well as references for additional learning. It was challenging to implement my dataset because I had grayscale images 48*48 pixels and had to convert them to 3D tensors. Both pre-trained models were trained using 224*224 color images. And, it was faces and other objects. Maybe, face specific models like ***VGGFace*** would be better choices and I will definitely try them in the future. I've used ***data augmentation*** here as well. My scores were lower here compare with the regular neural network. And ***VGG16*** performed better than ***Resnet50***. Since my dataset differs a lot from the dataset that the original model was trained on, it is better off using only the first few layers of the model to do feature extraction, rather than using the entire convolutional base and apply the ***fine-tuning*** techniques in the future.


### 4. Neural Style Transfer Notebook

 This notebook more as a learning of ***neural style transfer*** rather than my own discovery compare with other notebooks. This technique was first described by Leon Gatys in his publication 'A Neural Algorithm of Artistic Atyle' in 2015 and later by Keras creator and Google AI researcher Francois Chollet in his book 'Deep Learning' in 2017. Updated version and some modernization to the code were made by Baptiste Pesquet in 'Machine Learning Handbook' in 2021. All links to sources are included in the notebook.




## Conclusion:

  I've placed my 'local' conclusions at the very end of each notebook describing in details scores of my models and different configurations, the best libraries to import and work with images, additional thoughts and ideas for future improvements and developments. Although, my pre-trained models didn't perform as well as a regular convolutional network. I strongly believe some improvements could be made here. All operations were made on my local machine that slowed down my process of training for deep conv models. But, moving to the cloud would definitely help in improving the accuracy and loss scores because everything could be faster, and I could train my models for a larger amount of epochs.

 Facial emotion recognition needs to be researched and high-class models  to be developed every time. Many useful applications  have been already implemented based on FER in people's life such as temperature detectors, health monitors, security  techniques in airports and many more! I truly believe that car manufactures like Tesla, Toyota or especially electric truck  companies could install systems to their vehicles to monitor driver's faces and alarm them if they are fatigue or could fall  asleep to prevent car crashes.  It is our future. It would be used more and more often, and I hope only for good reasons! 
