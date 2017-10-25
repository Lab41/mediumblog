# Introducing source-contrastive estimation
## Learning deep embeddings for audio source separation

<img src="images/1.jpeg" alt="image of water bubbles">

Have you ever been talking with someone in a room crowded with dozens of people talking at full volume and been amazed that you can still hear and understand your conversation partner? I sure haven’t — it’s part and parcel of being a human with at least average hearing ability. We are remarkably effective at navigating noisy environments, and understanding what we hear in them. For the most part, however, it doesn’t even seem remarkable to us.

Listen to an audio recording made in the same environment, however, and you will realize why speech and engineering researchers have been stumped by the “[cocktail party problem](https://en.wikipedia.org/wiki/Cocktail_party_effect)” ever since the early days of signal processing. If you don’t have your two ears (and whole body, which also conducts sound) in the scene, the reality of a noisy environment like a bar or loud party is one of an undifferentiated din of overlapping speech, reverberation, and interference of all sorts. Recordings made in such environments are often unusable in their original state:

> [Sample audio](https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/324319477&amp;color=%23ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false&amp;show_teaser=true&amp;visual=true)

###### Source: [LibriSpeech](http://www.openslr.org/12/) Corpus

The field of source separation seeks to make signals like this usable by separating speakers onto separate tracks, recovering the original, much more intelligible, components of the mixture. Notice how the individual components of the mixture in Figure 1, on the right, have more orderly harmonics (the really narrow horizontal bands stacked on top of each other) and clearer structure in the vocal tract resonances (the thicker dark bands), compared to the mixture on the left:

<img src="images/2.png">

###### Figure 1. Left, spectrogram of mixed audio. Right, individual components of the mixture. Audio credit: [LibriSpeech](http://www.openslr.org/12/) Corpus

In the past few years, deep learning has begun to be applied to audio source separation. Such work includes Huang et al.’s [RNN-based approach](https://arxiv.org/abs/1502.04149), [innovative cost functions](https://arxiv.org/abs/1607.00325) from Microsoft Research and Aalborg University, and a vector-embedding technique adopted by [deep clustering](https://arxiv.org/abs/1508.04306) and improved on in [deep attractor networks](https://arxiv.org/abs/1611.08930). These techniques have typically added a large improvement over linear matrix decomposition techniques, like non-negative matrix factorization. We plan to put up a blog post in the near future describing some of those approaches. Today, however, let’s dive into Lab41’s unique work in this space, an approach we call [source-contrastive estimation](https://arxiv.org/abs/1705.04662) (SCE).
