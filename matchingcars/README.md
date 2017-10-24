# Matching Cars with Siamese Networks

![image showing cars on a street](images/matching-cars-1.jpeg)

Lab41 just finished [Pelops](https://github.com/Lab41/pelops), a vehicle re-identification project using data from fixed video cameras. Last time I talked about “chipping”, that is extracting an image of a vehicle from a frame of video automatically. We found that background subtraction worked OK based on the small amount of labeled data we had.

In this post I’ll go over the rest of the pipeline: **feature extraction** and **vehicle matching**.

## Feature Extraction
Machine learning algorithms operate on a vector of numbers. An image can be thought of as a vector of numbers — three numbers to define the color of each pixel — but it turns out that taking these numbers and transforming them gives a more [useful representation](https://en.wikipedia.org/wiki/Visual_descriptor). This step of taking an image and creating a vector of useful numbers is called feature extraction. We performed feature extraction using several different algorithms.

Our first method of feature extraction was an old one: Histogram of Oriented Gradients or HOG. HOG was first proposed in the 80s, but has since found uses in identifying pedestrians and was used by our sister lab [CosmiQ Works](https://medium.com/the-downlinq) to [identify boat headings](https://medium.com/the-downlinq/histogram-of-oriented-gradients-hog-heading-classification-a92d1cf5b3cc). HOG effectively counts the number and direction of edges it finds in an image, and as such is useful for finding specific objects. HOG lacks color information, so in addition to the output from HOG, we appended a histogram of each color channel.

Our second method of feature extraction was based on deep learning. We took [ResNet50](https://arxiv.org/abs/1512.03385) trained on [ImageNet](http://www.image-net.org/), removed the fully connected layers, and treated the 2048-dimension output of the convolutional layers as our feature vector. [It is well known that networks trained on ImageNet, despite being exceptionally good at identifying dogs and cats, are also very good for general image problems.](https://cs231n.github.io/transfer-learning/) It turns out the edges, shapes, and colors learned for dogs are also, in different configurations, useful for cars. For more details on the ResNet architecture, see my [reading group blog post](https://gab41.lab41.org/lab41-reading-group-deep-residual-learning-for-image-recognition-ffeb94745a1f).

Our third method of feature extraction was a [fine-tuned](https://cs231n.github.io/transfer-learning/) ResNet50. Pretrained networks are good at general image tasks, but they can be “fine-tuned” to perform better on specific tasks. For Pelops that specific task was make, model, and color identification of cars in a labeled dataset. It is hoped that making the model better at make, model, and color detection will generate features that are more useful for matching cars. This makes intuitive sense: any human matching cars would use make, model, and color as primary features.

## Matching
Once all of the vehicles have feature vectors associated with them, those vectors can be used to match vehicles to each other. There are a few ways to do this, starting with the simplest, which is to calculate a distance between the vectors. This works great if the feature extractors are designed to make the distance meaningful, but this is not generally the case. Neural networks can have cost functions based on distance, but ResNet50 does not. So this method, while attractive in simplicity, is not a good solution.

The second possible way of matching is to train a traditional (that is, non-deep learning based) classifier. We trained a logistic regression model, a random forest, and a support vector machine on top of each of the types of feature vectors. Each model was given two feature vectors and asked to classify them as coming from the same vehicle, or not. The training data was balanced so that there were as many positive as negative examples. The best accuracy these models achieved was 80%, although most struggled to pass 70%. Accuracy is the number of true results divided by the number of total items tested.

The third method of matching was to use a neural network as a classifier. Once we added a deep learning classifier on top of our deep learning feature extractor, we had a [Siamese neural network](http://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf). For details about how we trained such a network, and for an overview of its architecture, see our [blog post here](https://gab41.lab41.org/tl-dr-turn-static-portions-of-your-network-into-a-lookup-table-numerically-rank-your-architectures-46cc7750cad6). The Siamese network performs the feature extraction and matching in one step, and so allows optimizing both portions at the same time. This arrangement achieved the best results by far, hitting nearly 93% accuracy on our test set.

![A cartoon of our Siamese network architecture. The two convolutional blocks (CNN) output vectors which are joined together and then passed through a set of fully connected (FC) layers for classification.](images/matching-cars-2.png)

![Two example images from VeRi showing the same truck passing two different cameras.](images/matching-cars-3.jpeg) ![](images/matching-cars-3b.jpeg)

![Comparison of three matching methods with a random baseline using a cumulative matching curve.](images/matching-cars-4.png)
