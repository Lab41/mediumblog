# Sampled Softmax
## Training Deep and Wide Neural Networks Fast

You might have heard of this thing called deep learning, and maybe even understand how computationally complex it is. This complexity comes about because you have linear algebraic operations with matrices that are really big. It's a familiar theme: matrix multiplications are the bottleneck. It's a major reason why performance in BLAS, LAPACK, and signal processing libraries have been squeezed through generations of electrical engineers and computer scientists, We'd recently gotten even better at it with more specialized hardware like GPUs and [TPUs](https://cloudplatform.googleblog.com/2016/05/Google-supercharges-machine-learning-tasks-with-custom-chip.html). Updating matrices with [SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) is a core requirement for backpropagation in deep learning...but, what if you didn't have to update all the columns of the matrix all the time? What if I told that you that in deep learning, your largest matrix update could be computationally reduced a thousand fold and produce more accurate results in fewer iterations? What if, by doing this, you can train an image classifier to recognize millions of things in less time than it takes to train current algorithms to recognize thousands?

We recently stumbled onto the fact that you can do backpropagation, at least in the final layer, with drastically smaller matrix updates by sampling the labels. When you do this, not only will matrix updates be quicker per iteration of SGD, not only will it require fewer numbers of iterations, but you will even *improve* your accuracy while recognizing a ton more things. In this post, we'll talk about how exactly this happens using a classic image annotation problem, and we demonstrate that it can work on the most unstructured and noisiest of data: photos in the wild from the internet.

### The basic idea

It all starts with the fact that deep learning earns at least part of its keep with randomness: random initialization, random dropout, (hehe, random choices of architecture, random trying stuff out), etc. Closely related to randomness is the idea of sampling methods, where you're taking random points from data. 

If you have trained image classifiers, you've done it by randomly sampling your images and putting them in batches. Our contribution is the same idea but done to the max. Within each data sample, we're also going to sample the labels. If you take anything away from this blog post, it's that idea. Take a look at the animation below. We're first sampling our images, and then we're sampling the labels. As it turns out when you do that, you burn through your training much faster and more efficiently...and interestingly enough, achieve higher accuracy.

![Sampling Images](images/samplelabel.gif)

We really only tried the sampling stuff on the final weight matrix, mostly because we're assuming the classification layer is the most computationally burdensome...not an unfounded assumption. It's almost certainly the case if you're dealing with user generated content from the internet. These are photos that people have uploaded and tagged, and they can, and pretty much always do, tag them with whatever random word that comes to mind. You can imagine the number of possible things supervised machine learning can classify using these tags...in the millions in our case. Unfortunately, if we're using neural networks, that means that the last weight matrix is `hidden_units` $$\times$$ millions. That's a pretty big matrix...and that's the reason why our sampling method is necessary.

### Background: internet photos

Ah, open source multimedia; it's what 90% of the internet is made of. And while the internet is a dirty, noisy place, maybe there's something that machines can learn from it. One of the largest datasets that is composed of uncurated, open source multimedia is the [YFCC 100M Dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=i&did=67). The underlying reasoning for its initial release was its scale: lots of images, lots of metadata. A related issue, though, was its noisiness and overall difficulty to work with. It's easy to understate this, so there's nothing like actually looking at the data. If you actually parse through the corpus, you'll start to notice individual metadata tags sometimes make sense but oftentimes don't. For example, take the below.

![FlickR Creative Commons from https://www.flickr.com/photos/guenterleitenbauer/17272171332/in/photolist-UbULQ-oJxdAu-4ZhXEf-9UN1Bi-3m4YQy-sjhoMs-73J38d-j1dBZc-5Jv22Z-dYwK6H-Ldqa7-6gc4CL-7cXAsP-5B37Sm-FB9Hp-JbvZxC-pHgXDN-oCUpJ9-7RLduZ-eAW16B-poUzEv-6Q2wNH-bSyyp8-atdXmp-JjGiS4-oTGsxq-515ypv-HUwTYF-8YFaai-gmHyuJ-88pmb5-bECPMC-w4urw-JnPeit-HyNqxQ-hMrztt-f4hgYU-fdvTWF-4TMi9v-HvpJxY-gv1asQ-pRpjEW-HT8SbU-qjLVSo-bqDQCz-8vzYc4-bRrtNT-s6gE7k-7LtUp1-dfDc7T](images/venice.jpg)
```
Metadata Tags: 2015, April, Austria, Canon, Guenter, Günter, Landscape, Leitenbauer, Wels, bild, bilder, canal, canale, city, flickr, foto, fotos, image, images, kanal, key, landschaft, photo, photos, picture, pictures, stadt, town, venedig, venetia, venezia, venice, wasser, water, www.leitenbauer.net, Österreich, burano, island, insel, isola
```

Yes, we'll see `canal, venice, water, architecture`, which perfectly represent the picture. But the other stuff is enough to throw you off. I'm guessing that this was an Austrian Tourist who has a Canon camera, and he went to Venice in April 2015. To get a machine to recognize that is difficult, but that's another story altogether. The YFCC chalenge has been to be able to decipher *content* from an image, and in that sense, the tags `2015, April, Austria, Canon, flickr, foto, fotos, isola` are noise.

That's actually a pretty good example of the tags. There are some that are just inexcusably bad.

![FlickR Creative Commons from https://www.flickr.com/photos/imageborder/11741277024/in/photolist-qTvDJZ-oU4SDj-nVupEh-jEaNgS-nNJP5X-nSg7Cd-oytXam-eob2s2-opNMeb-oxqRfq-og44rm-njrx4M-9VPiN2-o2RKTf-s6uFUq-prHm2P-mvh2GK-omZVbp-huKcCJ-pQZATK-pgFPud-piZaUu-afEMB6-dx3181-ndVey3-zE74Jc-hTGDYX-nEqUsQ-nF4s6X-pfj2yc-o7Ursu-bxbP6B-pizidR-s6oC1x-nSPcU4-paTU56-oCHYpD-hu5wwJ-otgsJy-75fkX2-p1muFT-pdgnp9-8Dq8ZR-iTx7AL-o15sFG-mRseLM-aJuaiM-aD8pKX-owydez-qzaqsZ/](images/waves.png)
```
Metadata Tags: BlinkAgain, my_gear_and_me
```

This is literally the majority of the YFCC dataset; crap like this. It's bad, yes, but the hope is there is enough data to be able to overcome these issues. And, in fact, as you will see, there is...surprisingly so. Again, it's the idea that the individual tag and image will likely not make any sense, but the corpus as a whole will produce a pretty reasonable classifier.

### Our deep learning approach

Deep learning have mostly been trained with only thousands of labels on heavily curated, iconic, single-labeled words. Something like the YFCC dataset, though, has millions of labels. I ended up cutting it off at 400,000, but even that many words is difficult to accommodate on the GPU. 

![Large Neural Network](https://raw.githubusercontent.com/UCKarl/UCKarl.github.io/master/_posts/academic/images/largenn.jpg)

What *is* good at large vocabulary unstructured text are word embeddings, though. These have been called neural networks, but you should probably know that they're more wide than deep. One of the more well-known algorithms is *word2vec*, and in the subsections below, we’ll tell you how we adapted its most prized contribution, sampling, to our problem.

#### Word2Vec

In the past few years, not many papers have had more impact than Tomas Mikolov's [*word2vec*](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf). The major takeaway from *word2vec* is their use of negative sampling, through a broader idea of noise contrastive estimation. Here the negative in negative sampling just means that you’re sampling from the distribution of *unrelated* words. The function as written in his paper is:

$$ \max_{v_i, v_o} \log \sigma( v_o^T v_i ) + \sum_n \mathbb{E}_{n\sim p(i)}\left[ \log \sigma (- v_n^T v_i) \right] $$

Here, $$v_i$$ and $$v_o$$ are the input words and context words, respectively. To define an input word’s meaning, you’re using its *context*: its surrounding words (the first term). But you’re also defining the input word by what it’s *not*: things that *aren’t* in its context. That’s the second term, and the $$n \sim p(i)$$ under the $$\mathbb{E}$$ is a fancy way of saying you’re taking random samples $$v_n$$ over all words: i.e., negative sampling. 

The first term pulls words together (maximizing correlation) while the second term pushes them away from negatively sampled words. It can be shown that if the sum in this term extends to the entire word vocabulary, the above equation is equivalent to the cross-entropy function. (See [this](https://gab41.lab41.org/anything2vec-e99ec0dc186#.ddnjxweeq) for a visual description and our [ArXiV paper](https://arxiv.org/abs/1611.06962) for a mathematical one.) Instead with sampling, you’re approximating the distribution rather than empirically considering the entire word vocabulary. 

#### Extensions to im2vec

Again, *word2vec* uses an input’s surrounding words as context. In our case, the *context* of an image in the YFCC dataset can be its tags in the metadata, and the negative samples are unrelated tags. Most entries into image classification competitions use cross-entropy as the objective function, and as the *word2vec*` approximates the cross-entropy, it makes sense to use the exact same idea.

If you take a look at the previous section's equation, all we're going to do is replace the input vector with a neural network hidden layer feature. Then, we do backpropagation through the rest of the deep network.

$$ \max_{\{W\}, \{v\}} \sum_p \mathbb{E}_{p \in \text{tags}_i} \left[ \log \sigma( v_p^T g(x,\{W\}) ) \right] + \sum_n \mathbb{E}_{n \sim p(i)} \left[ \log \sigma(-v_n^T g(x,\{W\})) \right] $$

Here, the $$g(x, \{W\})$$ is our neural network with the set of weights denoted by $$\{W\}$$, which we are optimizing, and $$v_p$$ and $$v_n$$ are the context and unrelated tags, the positively and negatively sampled vectors. It might seem daunting at first, but all we did is replace our $$v_i$$ with some deep learning layers, and are now also sampling the labels because we have to do tensor algebra (the stuff in the first expectation).

[comment]: <> (Since we brought up word vectors, it's worth noting that lots of people have attempted to extend the use of word vectors to images with [hierarchical ontologies](https://github.com/li-xirong/hierse) and [semantic transfer](https://arxiv.org/pdf/1604.03249.pdf) and ["fast" zero-shot tagging](https://arxiv.org/pdf/1605.09759.pdf), but no one has seemed to do so at scale on UGC/open source multimedia. By scale, I mean both number of images and number of tags. For reference ImageNet has thousands of labels, YFCC100M has millions. Still, using the simple concept of negative sampling that Mikolov promotes, it's actually not quite difficult of an extension apply to images. There are some caveats, though, one of them being what hardware you're using.)

#### Sampling distributions

So, why negative sampling? With ordinary backpropagation, you're weighting all the unrelated stuff the same. By sampling, we're utilizing the actual probability distribution of the data itself. It's especially useful for internet data because there, the distribution actually matters. On *flickr* at least, there are a whole host of tags that are meaningless and come up all the time. A lot of them come about because people tend to turn autotagging on so things like `square format`, `nikon`, `instagram app` tend to turn up. We can all agree those are useless. (On a side note, I learned that `iphoneography` was an artform of iphone photography.) These will only be pushed away from image content based on sampling properties.

[comment]: <> (Meanwhile, positive sampling of images, where we sample according to a scaled inverse of the distribution, will pull images toward less frequent words. But because we're using the distribution of words, words that apply to a wide variety of images will be marginalized.)

[comment]: <> (There is that pesky issue of scale. If you've got 600k unique words, your output matrix will be of size 600k, and if you've got a second to last layer at 4096 dimensions, then the dimensionality of that matrix will be $$600k \times 4096$$, a pretty large matrix to backpropagate.)

Another thing we experimented with is positive sampling on the tags themselves. That is, you’ll want to consider the probability of occurrence for frequently occurring words (like `car` or `tree`) over stuff like `grandmother_kerri_janning`. The words in YFCC100M follow a [zipf distribution](https://en.wikipedia.org/wiki/Zipf's_law), and  there is a balance that we haven’t struck just yet for positive sampling. We just used uniform sampling for now. Please let us know if you've tried something that works better!

### The trouble with GPUs

You might've heard that GPUs are limited in memory. In all actuality, they've gotten pretty beastly in this respect. Also consider that with NVLink, memory sharing could fit tons of data. Still, tags in UGC can number in the tens of millions, especially without constraints on language or spelling errors. Without doing an excessive amount of coding, and if you're a poor researcher with a GeForce 700 series card, you may need an alternative.

Mikolov's work is all done on multi-threaded CPUs, with good reason. The motherboard simply has way more memory. Secondly, he deals with only wide neural networks, which means that optimizing a single layer in parallel may be just fine. It's analogous to HOG-Wild, where you're just randomly optimizing columns. The difference is that the chance of writing over a word vector that's being updated simultaneously is fairly low, since the vocabulary is really large.

My takeaway here was to use GPUs for deep learning. Use CPUs for wide learning. And in our case where we're training against internet photos and tags, we've got to use both GPUs and CPUs. That's because we need deep learning to deal with the complexity of images and wide learning to deal with a really large vocabulary. Incidentally, it’s also a gimmick to use the buzz term [deep and wide neural network in Tensorflow](https://www.tensorflow.org/versions/r0.11/tutorials/wide_and_deep/index.html).

### Some image retrieval for you

So we trained on *flickr* data...lots of it. And then, we did some example image retrieval by querying random words and seeing what we came up with. Mind you, the vocabulary in YFCC100M is ginormous and you can choose any word you want to search for. (There was a few exceptions when we presented at GTC; someone yelled out "Brexit", and since our data scrape was in 2014, it wasn't in our vocabulary.) 

![image](https://raw.githubusercontent.com/UCKarl/UCKarl.github.io/master/_posts/academic/images/yfcc-visualize.png)

In the above, our search terms are on top, and the images returned are on the bottom. The original tags are also included, and you can see that most of them make no sense. It's why it's surprising that our approach worked at all! Just goes to show you that there's enough signal in open source content that training deep learning is possible. You can see more of these examples in our paper and if you download our code. We'll provide models if you are interested.

### And there's much more...in code, papers, and repos

Thanks for reading this far. We're really excited about the implications of this work. Most of it was done at Lab41/IQT, though initially started at Lawrence Livermore National Laboratory. It's been presented at [CASIS](https://casis.llnl.gov/content/pages/casis-2015/docs/pres/Ni_CASIS2015.pdf), [GTC](https://disthost01.ovationevents.com/_NvidiaGTC-DC16/Wednesday/Polaris/dcs16133-karl-ni-vishal-sandesara-image-retrieval.mp4), is on [ArXiv](https://arxiv.org/abs/1611.06962) and submitted to CVPR. Most of it was done in under six months, but it was enough to serve the demo seen at GTC. You can access all of the Tensorflow code with the corresponding Docker Containers at the [Lab41 Github Page](http://github.com/lab41/attalos). For code we'd written to load in features easier, checkout our [CVPR submission](http://github.com/lab41/cvpr). For simple scripts that demonstrate the concept, you can peruse our [anything2vec code](https://github.com/Lab41/Blogs/tree/master/Anything2Vec).  If you have any trouble, feel free to reach out to me via [e-mail](mailto:kni@iqt.org), [twitter](http://www.twitter.com/karllab41), and my [website](http://uckarl.github.io).
