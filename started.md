---
layout: post
title: "Getting Started With Neural Networks"
permalink: /started/
date:   2017-01-11 18:15:11 +0530
categories: jekyll update
comments: true
---

## Introduction

Image recognition has emerged to be one of the most important applications of Artificial Intelligence owing to its powerful use in both social and scientific applications. Machine learning tools are being efficiently applied on images to extract information and make processing more human-like. With the advent of powerful machine learning algorithms and modular libraries like Keras for applying neural networks to such tasks, people have come up with innovating techniques to apply such tools to image processing and recognition. Kaggle offers a variety of problems on Image recognition and for beginners in machine learning, we will take an insight into two of the most straightforward and famous problems on Kaggle. These problems will help in getting to know how to apply machine learning algorithms and Neural Networks to Image recognition and processing task.

What you need to know

1. Knowledge of python is required as the code will be written in python and will extensively use python data structures like lists,dictionaries etc. Some amazing sources are available online for learning python.
2. Basic linear algebra like matrix and its operations

What you should have

1. Python installed and running. Ananconda is recommended as it comes with many packages already installed
2. Keras library installed and running. This will be used for Neural Networks.
3. IPython Notebook is a good choice for running scripts as it allows scripts to be converted into independent entities.

Neural Networks

According to Wikipedia,

```In machine learning and cognitive science, artificial neural networks (ANNs) are a family of models inspired by biological neural networks (the central nervous systems of animals, in particular the brain) which are used to estimate or approximate functions that can depend on a large number of inputs and are generally unknown.```

Neural networks are analogous to the way in which our human brain works. Consider the example when a child starts learning to walk. The baby first observes other people how they tend to take steps and then try to imitate them. On doing so he makes quite a lot of mistakes and tries not to repeat them by changing the style in which he/she walks. This is exactly the same way how Artificial Neural Networks approaches towards a problem. We start by feeding our model a significant amount of data. The structure of the data can be of any type but one regularity should be present. The data that is fed should be in the form of (x,y) pairs such that ‘x’ is the problem to be solved and ‘y’ is the corresponding “right” answer. The Neural Networks starts with some ambiguous parameters (say 0) and then changes its parameters so as to reduce the error produced for some new data point (say xi). The network keeps doing this until the error has been reduced to some value. The difference between simple regression and Neural Networks is that Neural Networks employ regression at quite a lot of different levels to achieve the desired accuracy.

## Implementing a Basic Neural Network for Handwritten Digit Recognition

### Getting the Data

Handwritten digit recognition is one of the most famous problems in the Deep Learning community and is considered as a starting point for anyone who wants to get their hands dirty in this field. The problem is to recognise the handwritten digit in 28×28 pixels greyscale image. By applying the neural network approach one can easily build a model to determine the number present in the greyscale image to accuracy as high as **99%~100%**.

The dataset can be downloaded from the Kaggle website. The dataset (also known as MNIST Dataset) consists of three files namely samplesubmission.csv, test.csv and train.csv

* test.csv : The file on which the model needs to be trained and outputs obtained are to be submitted

* train.csv : The file which provides the “training data”. This data is used to train the model to make it learn how to predict the number in the image according to the pixel arrangement

* samplesubmission.csv : This file instructing the format in which the data is to be submitted.


You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

### Knowing the Data

The data consists of series of rows and columns wherein each row corresponds to an image  and each column corresponds to a feature (or pixel). An Image is a row vector having dimensions of 1×784 i.e. a single image with 28×28 dimensions.

![Input image]({{site.url}}/img/number.png){:class="img-responsive"}

The above image is represented as a row vector as follows:

![Dataset image]({{site.url}}/img/data.jpg){:class="img-responsive"}

Our neural network will read the train.csv and convert it into a matrix where each row of the matrix corresponds to an image in the dataset. This matrix will be further used by our Neural Network based model to train its parameters and produce definitive results.

### Preprocessing of the Data

The data we have obtained is in the form of a .csv which contains comma separated values. CSV files are nothing but analogous to a spreadsheet in MS Excel or Numbers in OSX. Here each row of comma separated values corresponds to an image and each comma separated value is a pixel of an image. We will use the Pandas library in python to bring the CSV file into the memory

{% highlight python %}
import pandas as pd
import numpy as np
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt
%pylab inline

training_data = pd.read_csv('/path to your file/train.csv')
{% endhighlight %}

The training_data holds the training data into a Pandas.Core.Series format. We can use various methods to get more insights into our data like

{% highlight python %}
training_data.info()
training_data.describe()
{% endhighlight %}

Conventionally while creating the data for training a neural network, the data is divided into two parts. One of the parts is used for training the data and the other part is used for measuring the performance of the trained data. We use the ‘iloc’ function of pandas to bring about the first 30000 data points as our training example and the remaining points in our cross-validation dataset.

{% highlight python %}
X  = (training_data.iloc[:30000,1:].values)
X_cv = (training_data.iloc[30000:,1:].values)
{% endhighlight %}

Since the data fed to the neural network should be of the form (x,y) where x is the data point and y is the corresponding “correct” answer to the corresponding point. Looking closely at the data we got in train.csv we find that the first column corresponds to the labels for the data. Each row vector corresponds to the image and the first column of the determines which number is depicted in the image. Thus we separate this column out and store it in a variable

{% highlight python %}
labels = training_data.iloc[:,0].values
y = np.zeros((30000,10))
for i in range(30000):
    y[i][labels[i]] = 1
y_cv = np.zeros((12000,10))
for i in range(12000):
    y_cv[i][labels[30000 + i]] = 1
{% endhighlight %}

Finally, the data label is converted into a one-hot vector representation. One hot vector represents a vector in which all points except the label are deemed as zero. The label matrix looks like as follows

{% highlight python %}
array([[ 0.,  1.,  0., ...,  0.,  0.,  0.],
       [ 1.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  1.,  0., ...,  0.,  0.,  0.],
       ..., 
       [ 0.,  0.,  1., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 1.,  0.,  0., ...,  0.,  0.,  0.]])
{% endhighlight %}

### Creating a Neural Network Model (using Keras)

Since we have converted our 28×28 pixels images into their corresponding matrix representation using Pandas, it is time to create our Neural Network model. We will be using Keras to create and deploy our model. There exists a plethora of libraries for creating  Neural Networks of which Keras is the simplest owing to its large number of functions for each network specifications. Simple Neural Networks in Keras are represented with the help of sequential layers wherein each layer (comprising of various attributes) corresponds to a layer of the neural network. Consider the image below

![Neural Network]({{site.url}}/img/neural_net.jpeg){:class="img-responsive"}

The first layer is the input layer which takes into the matrix corresponding to our training set. The second and the third layer corresponds to the hidden layer wherein all the computation is performed. Each line joining the subsequent layers corresponds to a “weight” which is changed as the computation/training progress. Finally, the last layer is responsible for output which in our case is a matrix of one-hot vectors where the position of the 1 signifies the digit which is recognised.

The beauty of implementing a model in Keras is that one doesn’t need to worry about how to implement each of the above layers. The backend of Keras recognises and constructs the weight vectors automatically. All we need to give our model is the training set and the dimensions of our input matrix. Specifications like which error function to use and the number of epochs for training are specified while constructing the model. Now let’s try and construct our model using Keras (make sure Keras is installed and up running… )

{% highlight python %}
## Keras Model for implementing the Neural Networks
model = Sequential([
        Dense(32,input_dim=784),
        Activation('sigmoid'),
        Dropout(0.25),
        Dense(32),
        Activation('sigmoid'),
        Dropout(0.25),
        Dense(10),
        Activation('sigmoid'),
    ])
{% endhighlight %}

The above code snippet creates a Sequential model having three layers (input layer, a single hidden layer and an output layer). Each of the layers has activation function as the sigmoid. The activation function is the mathematical function which defines how the neurones from one layer are connected to the other layer. For the sake of simplicity models which require classification of objects (here Images) uses sigmoid as the activation function whereas in models which require a floating/integer output (eg. the price of a house subjected to change in socio-economic conditions), the activation function ‘relu’ is considered. You can look at the different activation functions present and their corresponding usage on the Keras documentation. We use a 25% Dropout rate so as to prevent out model from overfitting. Overfitting is one of the most common errors while implementing the machine learning algorithms. A model is said to have gone under overfitting when its parameters have been so adjusted that they seem to work very good on the training set but fails to predict with accuracy on the cross-validation set or any other data point outside of the training data.

#### Input Layer

The first layer as shown below has called to three functions, namely, the Dense, the Activation and the Dropout. The Dense function is responsible for constructing the neurones and the other two layers specify the attributes of that layer. The Dense function needs to be supplied with an argument (input_dim) so as to arrange for the length of the vector for each of the neurone specified. For our example, each image had a size of 28×28 pixels and was made into a vector of 1×784. Thus our input layer has 32 neurones where each can handle a row vector of 1×784. As we train our model our training set (matrix of the number of images x 784) supplies our model a row vector (1 x 784) one by one so as to train it for each epoch.

{% highlight python %}
Dense(32,input_dim=784),
Activation('sigmoid'),
Dropout(0.25),
{% endhighlight %}

#### Hidden Layer

{% highlight python %}
Dense(32),
Activation('sigmoid'),
Dropout(0.25),
{% endhighlight %}

The hidden layer is same as any other generic layer except for the fact that here we don’t need to specify the input_dim variable as it reconfigures the number of neurones in itself according to the layer preceding it (here input layer). Thus all we need to specify for Dense is the number of neurones in that layer and their attributes (Activation and Dropout).

#### Output Layer

{% highlight python %}
Dense(10),
Activation('sigmoid'),
{% endhighlight %}

The output layer needs to be supplied with only the “correct” number of neurones and the activation function. For our example, we need to classify the image as to which number it belongs. Since the only possible output is the set of integers from 0 to 9 (it is also specified by the MNIST dataset) so we need to have 10 neurones in our output layer such that each neurone is activated for a particular number to which the image belongs. If there would have been alphabets (only upper case) in place of the number then we will need to have 26 neurones in our output layer (for each character).

### Compiling and Fitting a Neural Network

Since now we have created a mathematical model of our data and also incorporated it into our model (created above), we can now train our model on the given dataset and use the weights that we obtain to obtain results for the given test set.

First, we need to compile our model by specifying the various functions that our neural network will use so as to compute the error and the gradient. With each “epoch” the model tries reduce the error by computing the gradient and changing the weights. The following line of code uses ‘categorical_crossentropy’ as the error function and ‘adadelta’ as the optimizer. You can read about these parameters more over here.
{% highlight python %}
model.compile(optimizer = 'adadelta', 
			  loss = 'categorical_crossentropy', 
			  metrics = ['accuracy'])
{% endhighlight %}

After compiling our model we need to train it on our training data. For this, we supply our model with the training set, the number of epochs (or the number of times) it trains the model on the training set and also the batch size (it determines the size of batches in which the training set is broken down and trained separately). The batch size shouldn’t be greater than the size of the training set or else it will be reported as a warning. Turning verbose to 1 will illustrate the training in progress as a progress bar (for ipython notebook better turn it to zero or it will sometimes show an I/O error). The following code snippet is used to train our model.

{% highlight python %}
hist = model.fit(X,y,nb_epoch = 50,batch_size = 32,verbose = 0)
{% endhighlight %}

### Evaluating the model on the cross-validation set

It is a good practice to break down the data into two parts namely, the training set and the cross-validation set. The model is first trained on the training set up to the desired accuracy and evaluated on the cross-validation set. The cross-validation set provides a way to check for how many datapoints our model was able to answer correctly on a dataset it hasn’t seen before. The metrics obtained on evaluating the model on the cross-validation set gives the true accuracy of the model and also tells if the model is overfitting the data or not.

{% highlight python %}
score = model.evaluate(X_cv,y_cv,batch_size = 32, verbose = 0)
print score
{% endhighlight %}

On printing the score, we get the following output

{% highlight python %}
[0.30793251706163088, 0.90666666666666662]
{% endhighlight %}

The 0.9066 represents the accuracy obtained which is approximately equal to 91%. That’s pretty good, eh!

We can also plot the graph of accuracy and the error (loss) as the training progresses. The following code snippets will demonstrate

{%highlight python %}
info = hist.history
plt.plot(info['acc'])
{% endhighlight %}

It will produce a graph similar to this,

![Accuracy graph]({{site.url}}/img/acc_graph.png){:class="img-responsive"}

{% highlight python %}
plt.plot(info['loss'])
{% endhighlight %}

It will produce the graph as follows,

![Loss graph]({{site.url}}/img/loss_graph.png){:class="img-responsive"}

The above graphs show that our training goes on pretty smooth. The error is decreasing as the training progresses and the accuracy increases smoothly as the epochs get higher.

### Bringing in the Test data and Predicting the output

Since now we have trained and evaluated our model, its time to put it into action. First, we need to bring the test data file into the memory. With Pandas it is simple,

{% highlight python %}
test_data = pd.read_csv('/path~to~your~file/test.csv').values
{% endhighlight %}

The `.values` method will convert the Pandas series into a Numpy array. Finally, we test our model on the test data and save the predictions into a variable yPred.

{% highlight python %}
yPred = model.predict_classes(test_data)
{% endhighlight %}

`model.predict_class` will automatically find the class with the highest probability. For eg, if the model predicts a probability distribution as [0.7,0.2,0.01 …… ] it will produce the output as 1 since the class having highest probability is 1.

### Saving the Predictions as a CSV file

Kaggle requires the predictions to be saved in a CSV formatted file. The format of submission may change and can be inferred from sample_submission.csv file that comes with the data. For this problem set, we can use the Numpy function to create a simple CSV file as required by the problem.

{% highlight python %}
np.savetxt('submission.csv',np.c_[range(1,len(yPred)+1),yPred],delimiter=',',header='ImageId,Label',comments='',fmt='%d')
{% endhighlight %}

Now you can submit the submission.csv and get your place on the leaderboard =)

{% include disqus.html %}
