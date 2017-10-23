# New Sampling in Tensorflow
## Contrastive Estimation in Practice

At a recent MMCommons event, we had the opportunity to sit in on a panel with computer vision scientists on dealing with on unstructured data. After giving a demonstration from our submitted work on open source image classification, it came as no surprise that the majority of the speakers piped up at length about how sampling methods have come to their aid. These amounted to various cases when the label space is too large or there wasn’t enough data. 

Separately and on the totally different domain of speaker separation, we reached the same conclusions when we implemented our very own source-contrastive estimation (SCE) algorithm, documented in another post by my colleague. We admit that it is a shamelessly gimmicky name (shh, don’t tell reviewers) for an enhancement on noise-contrastive estimation. But the idea remains the same: where you’re separating distributions from each other, i.e. contrastive estimation, sampling methods can be your ally.

![Sampling is for everyone](images/costco-sample.png)

---

How exactly does sampling help? Regarding sampling in the TensorFlow documentation, the phrase that you’re searching for (and the whole point of this post) is candidate sampling. The idea comes about when training a single example: we don’t want to go through the costly operation of evaluating every possible class, a big downer when you have a ton of labels. What you can do instead is build a function that only uses a small subset of your labels (i.e., sampling your universe of labels) to approximate the original “exhaustive” loss function, which may take the form of, say, the full softmax function.

The technique is fairly powerful and the idea is proven out from word2vec embeddings to Restricted Boltzmann Machines. But how do you actually do it? What are the relevant TensorFlow calls? Turns out there are some easy solutions that you may not have known about that could save hours of effort. In this post, we’ll take you through the algorithms and tools we’ve experimented with, some of which have made their way into our code and publications. If we’re instructive, then by the end of this article, you should be able to optimize for a sampled version of your novel cost function, or any arbitrary cost function for that matter, one using TensorFlow. 


---

### A Little History

The current libraries are documented with a lot of goodies that include cost functions on top of sampling distributions. For example, noise contrastive estimation is available:

```
tf.nn.nce_loss(weights, biases, inputs, labels, num_sampled, 
               num_classes, num_true=1, sampled_values=None,
               remove_accidental_hits=False, partition_strategy=
               'mod', name='nce_loss')
```

It wasn’t always the case that these guys were there for you though, and it surprises me how recently they’ve been added.  For example, stuff in `candidate_sampling_ops.py` has the last git pull just last month, and the documentation was uploaded in Dec 2015.  Back in the day, word2vec and most of the other label sampling methods didn’t have GPU support. In fact, look at the example on line 148 in TF 1.1; it still reads:

```
# Ops and variables pinned to the CPU because of missing GPU implementation 
with tf.device('/cpu:0'):
```

The lack of implementations of large layers was primarily a function of the fact that in cases where you would even need to sample the labels, the size of the weight matrices is proportional to an extraordinary amount of unique labels (e.g., in the YFCC corpus, it’s 400k words after it’s been pruned from six million words.) Memory on poor researcher GPUs had a hard time keeping up a few years ago.

![Perhaps some NVLink?](images/please-sir.png)

We experienced much the same, looking on in jealousy at the contributions from Google and Facebook. But then it happened…we finally got our Titan X cards. We got six of them, in fact, and then we were tickled to death. 
With this huge hammer, we looked around for the biggest nail. So late last year, we downloaded the biggest dataset we could get our hands on (100 million images) with the most amount of metadata (14GB), and threw the largest neural network we could at it. The trouble was, while we could find ways to fully use the 16GB of memory, the actual matrix-vector operation still took way too long. Enter candidate sampling, and why we’re here today.

### Canned Functions for Your First Pass

Again, a lot of the functions are semi-new. The TensorFlow community has been rushing to accommodate with generous contributions to the official repository. Specifically for candidate sampling, we’ve got all sorts of capabilities from `sampled_softmax` (this one’s an oldie), `nce_loss`, `tf.nn.uniform_candidate_sampler`, and all of their variants. You can see what all of these guys mean from the documentation, nicely summarized in:

![Table: Candidate Sample](images/candidate-sampling.png)

Logs and odds, oh my! All you really need to know is that the *positive* and *negative* samples (second and third columns) relate to the match or mismatch between the label and the data. The remainder of the table just tells you how to use those samples.

This is all great if what you’re trying to do falls into any one of these categories. The thing is, you might want to do something different, and sometimes cost functions don’t really fit into any of the above forms. Secondly, if you want to sample from a specific distribution, you’re beholden to the available ones in TensorFlow, which only include the *Zipfian* distribution, the boring and ever-present uniform distribution, and an empirical distribution (think histogram) based on a record of your past true samples. Things might have changed as of this writing, but that’s hardly a complete set of distributions.

### Messin' with Vectors

So…what if you want to do all that stuff yourself? What if you had no choice but to do the sampling manually because you needed to manipulate the vectors (embeddings) in some special way that doesn’t happen to be in that table above…and you wanted to do it *on the GPU*? In other words, we’d like to know how to sample off-GPU and then do math-stuffs on-GPU.

We found the most amount of flexibility in the function `gather_nd`, and implementing dot products ourselves.

#### A Little Drama

I’ll get to the explanation of what `gather_nd` does (and why you should use it) a little later, but first an odyssey of big data proportions. TensorFlow default installation (as of this writing) when you type in pip install `--upgrade tensorflow-gpu` is an older version 1.0. This little fun fact provided us with hours of ever-so-fun entertainment, which we didn’t know awaited us when implementing with `gather_nd`. It looked like we were going to make the SIPS and NIPS 2017 deadline, and then, que disastre! Irrecoverable errors nuked our graphs. Scouring the forums for an explanation, we almost lost our minds when we read this.

![Oh no](images/oh-no-1.png)
![Oh no](images/oh-no-2.png)

The nonchalant, “Yeah — we haven’t written the gradient implementation for gather_nd yet,” was enough to make our hearts skip a beat. What followed in the thread was a bunch of, “when will it get implemented?” and, “is it done yet?” The thread was so long, we left for home before reaching the end. Had we scrolled far enough, we would have seen this:

![Done](images/done.png)

This comment is April 6, 2017, but the implementation finished a few months before that. Whew, it’s a simple fix: you will need to get at least version 1.1, which you can do directly from the TensorFlow Github page. (They do nightly builds.)

#### Gathering and Broadcasting

The documentation for `gather_nd` is chock-full of examples, but the bottom line of code sums it all up:

```
sampled_vectors = tf.gather_nd(all_vectors, sampling_indices)
```

From a set of vectors (or matrices or tensors, whatever), you can extract samples from it however you like. And you can do this with all sorts of shapes! Let’s say I have an embedding that I’m trying to train up that’s of size 3 x 2, i.e. three examples, each with two dimensions. Try the below code out:

```
In [1]: import numpy as np; import tensorflow as tf
In [2]: Vo = tf.Variable( np.array([[0,1], [2,3], [4,5]]) )
In [3]: init_op = tf.global_variables_initializer()
In [4]: sess = tf.Session()
In [5]: sess.run(init_op)

# If I want the element in the matrix Vo at index (0,1):
In [6]: sess.run( tf.gather_nd( Vo, [[0,1]] ))
Out[6]:
array([0])

# If I want the 3rd vector in the matrix Vo (which is index 2):
In [7]: sess.run( tf.gather_nd( Vo, [[2]] ))
Out[7]:
array([[4,5]])
```

What’s really cool is that you might want a whole bunch of vectors to build a batch. You can repeat vectors a whole bunch of times:

```
# If I want the 3rd vector in the matrix Vo (which is index 2)...a bunch of times
In [8]: sess.run( tf.gather_nd( Vo, [[2], [2], [2], [2]] ))
Out[8]:
array([[4, 5],
       [4, 5],
       [4, 5],
       [4, 5]])
```

So that’s great, right? Almost. Let’s be clear: neural networks are all about tensor dot products, and navigating dimensions is especially hard. A typical situation occurs when you’ve gotten the samples (maybe from `gather_nd`) for a single input, and now you want to find the correlation to a feature vector a la word2vec. That is, you want to take a single vector and repeatedly dot it with a whole bunch of other vectors.

Turns out broadcasting works especially well, and the TensorFlow version of broadcasting works exactly the same as the numpy version. The element-wise `*` operator will replicate as needed, and you need only sum the relevant indices. For example:

```
# A feature and 3 vectors, dimensionality 2.
vi = tf.random_normal([2])
Vo_sampled = tf.random_normal([3,2])

# Correlation between vectors
corr = tf.reduce_sum( Vo_sampled * vi, axis=-1 )
```

You ask, what about multi-dimensional tensors? Neural networks are optimized by batches, right? As it turns out, when your dimension is of size “1”, TensorFlow will expand to the appropriate size. (Think `repmat` and `.*` for all you  MATLAB users.) For example,

```
Say I have a couple tensors of different sizes:
t1: 20 x 4 x 5 
t2: 20 x 3 x 5

Expand them (say with reshape or np.expand_dims)
t1': 20 x 1 x 4 x 5 
t2': 20 x 3 x 1 x 5

Voila, you can take the elementwise product!
t3 = t1' * t2', 
t3: 20 x 3 x 4 x 5
```

#### Putting It All Together

Let’s put it all together. Let’s say we have a batch of features. For each of the features we want to correlate with 3 vectors we’ve sampled from a set of vectors. (In reality, we’d sample 3 vectors 100 times for our SCE algorithm, but with the below code, the extension’s left out for argument’s sake.) This is how you would create that graph:

```
# Batch of features and a set of vectors to be optimized
Vi = tf.random_normal( [100, 2] )
Vo = tf.random_normal( [20, 2] ) 

# Sample 3 vectors from Vo
samples = np.expand_dims( np.random.choice(20, 3), axis=-1 )
Vo_sampled = tf.gather_nd( Vo, samples ) 

# Put a "1" where you want the vector to be replicated
Vi_expanded = tf.reshape( Vi, [100, 1, 2] )
Vo_sampled_expanded = tf.reshape( Vo_sampled, [1,   3, 2] ) 

# Calculate the correlation
corr = tf.reduce_sum( Vi_expanded * Vo_sampled_expanded, axis=-1 )
```

### Summary

That should be all you need to know to implement your own sampled cost functions. To review, we’ve linked you to candidate sampling. We’ve talked about a few of their cost functions. Then, for the fancier friends who enjoy sampling and implementing new cost functions, we explained TensorFlow’s `gather_nd` function to index into your embedding arrays. Then we talked about tricks in broadcasting, namely using “1” in places where you’d like to repeat operations, effectively creating copies of an array for operation.

We hope you enjoyed our TensorFlow tutorial. For more information, check out our Lab41 Github pages at: https://github.com/Lab41. Thanks for tuning in!
