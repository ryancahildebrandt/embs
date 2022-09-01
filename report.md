# The Ex-Academic's Sentence Embedding Guide

## Overview
This report seeks to serve as a quick reference guide for individuals working with natural language data, in tasks where it may be necessary to perform any of the following:

- Cleaning and preprocessing text data
- Representing text data in numerical format for computational analysis
- Identifying similar sentences
- Partitioning text datasets into groups of similar datapoints
- Visualizing semantic groupings/similarity
- Explaining or answering questions about any of the above to a non or less technical audience

While this report aims to be as comprehensive as necessary for most users and use cases, it should not be used as a substitute for a working understanding of the techniques discussed. Not every clustering algorithm is appropriate for every data type or every task, and using the right approach for the task in front of you is more important than having the most advanced or perfectly tuned algorithm. Because of the range of possible metrics and algorithms contained here, it is expected that some of the many, many possible combinations will be incompatible in practice, and fine grained selection of techniques should be done based on the specific data and task at hand.

This report was build alongside an app, hosted [here](https://huggingface.co/spaces/ryancahildebrandt/embshttps://huggingface.co/spaces/ryancahildebrandt/all_in_one_sentence_embeddings) on HuggingFace.co Spaces. The app allows you to play with the data in real time, combining different preprocessing, embedding, clustering, and dimensionality reduction techniques as you please!

## Getting Started
If you have a particular section you've come into this report interested in, feel free to jump right there. If you're looking to get a step by step walkthrough, I'd recommend going through the sections in order, as I've structured the report to cover the different phases of a sentence embedding/clustering task as you would likely do them in practice. Where possible, I've tried to include examples from a single dataset, though in some places where a particular technique may be useful appropriate examples are drawn from different datasets. To start, I'll use the "rec.autos" collection from the 20 Newsgroups dataset, accessed through scikit-learn. To make it a little more user friendly I'll also split the sentences by newlines to start, as the entries in the dataset consist of multi-sentence emails, as well as the sender and related conversation details

## Sources
The information here has been summarized from a variety of sources (in addition to some professional experience), and I highly encourage you to read more from those sources if you're interested in the finer details

- [NLTK](https://www.nltk.org/)
- [SciKit-Learn](https://scikit-learn.org/stable/user_guide.html)
- [HuggingFace](https://huggingface.co/models)
- [SentenceTransformers](https://www.sbert.net/examples/applications/computing-embeddings/README.html)
- [TensorFlow Hub: USE](https://tfhub.dev/google/universal-sentence-encoder/4)
- [UMAP](https://umap-learn.readthedocs.io/en/latest/)

## Preprocessing
Preprocessing is probably the simplest (and as a result the most taken for granted) step in any NLP pipeline, and is sometimes collectively called text normalization. Most of these steps will be best practice for a lot of different tasks, but it's always important to keep in mind which of them will have an impact on *your* data, especially if you're implementing all of this from scratch

### Lowercasing
Converting all characters in a string to their lowercase forms (where applicable)
- **Benefits**
	+ Helps to keep vocabulary size small (relevant to bag-of-words embedding algorithms), and makes sure that differences in case do not result in otherwise identical words being considered distinct
- **Useful Case**
	+ Lowercasing is useful in any case where you may have variation in capitalization
	+ This can be the case with written communications or less formal situations where representations of words are less consistent. For a lot of tasks, there's no point in considering "clams are the chicken nuggets of the sea" as different from "CLAMS ARE THE CHICKEN NUGGETS OF THE SEA" or "ClaMS aRe thE ChicKEn NUGgetS of THE SeA"
	+ Lowercasing may be less advantageous if you're not also implementing named entity recognition (NER), which tags proper nouns, names, and the like. In these cases, it may be helpful to consider words with differing capitalization as distinct, such as "my name is John Smith and I am a knife smith" in order to have a better capacity to parse the differing senses of the words
- **Example**

### Punctuation Removal
Removing all punctuation (including special characters and emoji depending on who you ask) from a string
- **Benefits**
	+ Depending on the embedding technique, punctuation could be considered as its own "word" within a sentence or be tacked onto the end of the previous word. Not only does this potentially cause issues in building the vocabulary, but can also throw off embeddings by considering punctuation as a meaningful entity within a sentence. In some sense punctuation *can* convey meaning (?, for example), but its inclusion tends to cause more issues than it solves in the context of sentence embeddings
- **Useful Case**
	+ Punctuation removal is useful in cases of text with *non-meaningful* special characters or inconsistent usage of non alphanumerics
	+ Since emoji, emoticons, and uncommon punctuation may be more common in text communications, these characters may unduly influence embeddings and throw off any subsequent analyses. For example, if you're looking to model topics within the utterance "i love coffee so much please send me 15 pounds of your finest donut shop immediately ☕☕☕ @@@@ 📫📫", considering 📫 as a topic probably isn't useful
	+ In the case of phone numbers or emails, however, characters such as @ or . may be useful to leave in place to assist in parsing these appropriately
	+ Keeping punctuation may also be useful in the detection of sentence or clause boundaries, which can help split sentences more appropriately
- **Example**

### Stopword Removal
Stopword removal is, as advertised, the removal of stopwords. Stopwords are broadly defined as extremely common words in a language whose inclusion contributes relatively little meaning or may be so common as to not distinguish one utterance from another. Some stopword lists may be context or field specific or even curated by an individual analyst, though many tasks will only use a language-specific, topic-agnostic stopword list
- **Benefits**
	+ The benefits of stopword removal align with stopwords' characteristics, removing stopwords keeps vocabulary size small and removes words which are unlikely to differentiate utterances and their meaning
- **Useful Case**
	+ Removing stopwords may be useful in cases where the given stopwords are non-informative in your data
	+ Cases where negation is not informative, such as when "I'm **not** calling about the birkins, they're in storage" should be considered the same as "I'm calling about the birkins, they're in storage", is a straightforward example, as "not" is an extremely common stop word in many lists
	+ The benefits or drawbacks of removing stopwords may vary greatly based on the specific list and data being used, so it's best to always check what words you're dropping
- **Example**

### Stemming
Stemming refers to consolidating all forms of a base word into the base word itself. As compared to lemmatization, stemming is more reliant on rule based algorithms, and tends to be more effective for simple prefix and suffix based inflections
- **Benefits**
	+ Stemming also keeps vocabulary size small and allows the embeddings algorithms to consider all variations on the base word to be the same. This can prevent two utterances with differing word forms from being considered different
- **Useful Case**
	+ Stemming has the benefit of being lightweight because of its rule based nature, and as such is good for implementations which need to be light and fast
	+ Stemmers handle rule based inflection very well, and are a good choice for languages or tasks where inflection rules are very regular
	+ The tradeoff of rule based stemmers is that they may apply their rules erroneously, which may result in nonsensical word changes (i.e. "hand me that painting" could become "hand me that paint", "I'm moving to a new organization" could become "I'm mov to a new organ")
	+ Different stemmers use different sets of rules, and may vary widely in their ability to recognize and parse names or proper nouns
- **Example**

### Lemmatization
Lemmatization is similar to stemming in that it aims to reduce word variant forms into their base form, but the approach used in lemmatization incorporates part of speech and other linguistic information. Modern lemmatizers are better at handling more complex inflectional patterns, but are more complex and generally slower
- **Benefits**
	+ The benefits of lemmatization are largely the same as stemming, though perhaps more pronounced due to the accuracy increase lemmatizers offer over stemmers
- **Useful Case**
	+ Lemmatizers are useful in most of the same instances a stemmers
	+ Lemmatizers tend to be trained on corpora, and therefore are more able to handle the natural variation in inflectional rules within a language as compared to stemmers
	+ Lemmatization generally performs better than stemming in most applications, and really the only drawback is the additional memory or processing power needed to implement lemmatization over stemming
- **Example**

### Spelling Correction
Spelling correction can use a range of algorithms of varying complexity, but the core task accomplished by all of them is to substitute misspelled words with the most likely candidate word
- **Benefits**
	+ In addition to keeping vocabulary size small, spelling correction ensures that embedding algorithms will not have to consider words that are misspelled
- **Useful Case**
	+ Spontaneous utterances (such as text message conversations) may be good candidates for spelling correction, either through rule based parsers or more computationally expensive approaches such as distance based corrections. Understanding that "don't frget the frosteed glakes please in eed them for the bool club" is likely the same as "don't forget the frosted flakes please i need them for the book club" can be important for text corpora with large variation in spelling accuracy
	+ Spelling correction may be less than ideal in texts with less common words or specialized terminology, where spellcheckers trained on high frequency words may not recognize or correct appropriately
- **Example**

### Clause Separation
Clause separation is not a standard preprocessing step in many pipelines, but can be useful in the case of longer sentences or sentences containing multiple topics. Sentences are split by a range of criterion, including punctuation and/or words commonly denoting boundaries between clauses or topics in a sentence
- **Benefits**
	+ Depending on the embedding algorithm, embeddings for longer sentences may not adequately capture the range of topics in a longer sentence. By splitting a sentence into clauses, the embedding algorithm can more accurately capture the meaning of each clause without those being beholden to the content of the whole sentence
- **Useful Case**
	+ Clause separation can be especially useful for sentences with multiple disparate topics, in order to classify the parts of such sentences into smaller clusters or more narrowly defined categories. Splitting "don't call this number again, i don't know what kind of vacuums you're selling but i know for a fact they can't handle the mess that is my life" into "don't call this number again", "i don't know what kind of vacuums you're selling", and "i know for a fact they can't handle the mess that is my life" may make the information more readily parsable by the embedding model
	+ It may not be ideal for tasks where sentences need to stay as provided, or the main topic is the only one which should be considered. 
- **Example**

## Embedding
At its most basic, an embedding (in the context of NLP) is the representation of semantic information as a vector of numbers. These number vectors can be treated as any other, acting as coordinates in high dimensional vector space. Word embeddings are in some cases a precursor of sentence embedding techniques, but will not be covered here in detail. There are any number of approaches to arrive at an embedding vector, but most approaches fall into one of two categories: Bag of Words models and Neural Network based models
- **Bag of Words Models** : Count, Hash, tf-idf
	+ Bag of words models encode the meaning of a sentence based on the words in the sentence and the words in the corpus
	+ Embedding vectors are constructed from the counts of each token (individual word or n-gram) in a given sentence, with each position in the vector corresponding to a unique token in the corpus
		* **Useful Case**
			- Bag of words models are useful when you need to maintain the interpretability of the embedding features. Because these models explicitly encode each word in a sentence, the embedding values can usually be mapped onto the original words
			- They are also good when you want to make sure that the embeddings are *explicitly* encoding certain words, not the broader meaning of those words. For example, if you want to make *sure* "lobster" is encoded as a distinct feature from "crab", even though they may be semantically very similar and used in similar contexts, bag of words models may be your go-to
- **Neural Network Models** : SentenceTransformers, Universal Sentence Encoder
	+ Neural Network based models, as the name suggests, use neural networks trained on large corpora
	+ Unlike Bag of Words models, they do not represent tokens directly in the embedding vector, but rather embed the meaning and context of a sentence into a fixed vector space, which will produce similar embeddings for sentences with similar meanings
	+ Because these models have been trained on specific corpora and in some cases for specific tasks, there is a wide range of models to choose from in approaching any NLP task, each with drawbacks and benefits of their own
	+ For brevity, the different models will be glossed over here in favor of a broader look at neural network embeddings as a whole
		* **Useful Case**
			- Neural network models are useful for capturing meaning in a deeper sense than the specific words included in a sentence. For example, a neural network model will likely be more able to understand that "how many pounds of butter is that" is more similar to "tell me how much margarine is here" than to "that is so many pounds of butter", even though the latter contains more of the same words.
			- By the same token, they can be less interpretable because the vector space does not map explicitly to features of the dataset, and instead encodes semantic features which are not easily identifiable or separable
			- They tend to outperform bag of words models in accuracy, but the tradeoff with interpretability may not be worth it in all cases

#### Count
Count embeddings are perhaps the simplest of the bag of words models, collecting all of the unique tokens in a corpus into a vocabulary array, and storing the counts of each token as the embedding vector
- **Benefits**
	+ The primary benefit of a count based embedding is interpretability. Because this algorithm stores the token vocabulary, it's simple to identify influential/differentiating tokens. Like all bag of words approaches, count embeddings tend to be lightweight and quick to implement due to the simplicity of the algorithm and the ability to limit the vocabulary size as needed
- **Example**

#### Hash
Hash embeddings are nearly identical to count embeddings, with the exception of not storing the vocabulary array and instead relying on position within the vector to map to a particular token. Depending on vocabulary size parameters passed to a hashing vectorizer, the results may be identical to the count vectorizer (with the exception of the vocabulary array)
- **Benefits**
	+ While they benefit from the lightweight and quick nature of bag of words approaches, hash vectors do not store the token vocabulary alongside the embeddings. This means its not generally possible to identify important tokens within the embedding, but because the token vocabulary doesn't have to be stored at all, there is more memory available for the embeddings, and as a result fewer potential constraints on vocabulary size
- **Example**

#### TF-IDF
The main differentiator of tf-idf embeddings is the "term frequency-inverse document frequency" weighting applied to each token count, which provides a measure of how "important" a token is to a particular sentence both in terms of the token frequency *within* a document (tf) and the token frequency *across* documents (idf). Via this weighting, tokens that are more likely to differentiate two sentences are weighted more heavily and less informative tokens are weighted less heavily
- **Benefits**
	+ The tf-idf weighting allows for an embedding which takes into account the importance of each token in addition to its prevalence. This can in many cases provide a boost to accuracy
- **Example**

#### Sentence Transformers & Universal Sentence Encoder
Universal Sentence Encoder and the models included via the Sentence Transformers package are all built on some variation of neural network (sometimes with fancy layers or nodes), trained on some corpus of language data (sometimes from a specialized area), and specialized for some specific task (or specialized to be good at as *many* tasks as possible, in the case of transfer learning specialized models). The pretrained models included in Sentence Transformers are hosted on HuggingFace Model Hub, where each model page explains model details and use cases. In the case of Google's Universal Sentence Encoder, the model and its different versions are hosted on TensorFlow Hub. TF Hub also lists use cases and example implementations, in addition to a range of models similar to Sentence Transformers
- **Benefits**
	+ The main selling point for any neural network embedding model will almost certainly be accuracy. By training the network to factor in individual tokens in addition to the context of the entire sentence, neural network models are able to capture much more nuance and embed them into a more complex vector space. This is, however, much more computationally intensive and relies on the pretrained model itself, which must often be downloaded, stored, and loaded in order to produce the desired embeddings
- **Example**

## Clustering
Once you have your embedding, the next step is of course to *do* something with it. Depending on your task, you might only need to identify a few similar sentences (semantic search), or you might need to return a likely response to a question (Q&A), or you might need to find patterns in your data by putting similar sentences together (clustering). While the embedding model used in the previous steps might be specialized for one or more of these tasks, we'll be talking about clustering algorithms here and can technically use any of the embedding models for clustering. Clustering algorithms are a form of unsupervised machine learning, where we aren't providing labels for the algorithm to match to, but instead setting general parameters that the algorithm will use to *find* patterns in the data. These parameters will vary from algorithm to algorithm, as will the basic approach used by each. Many of the algorithms below rely on a distance metric, which defines the distance between two datapoints. Each datapoint is defined by an embedding vector, which functions as coordinates in the embedding vector space. For a sentence embedding you'd be hard pressed to find a 2 dimensional embedding vector, but conceptually a 2 dimensional embedding vector (length of 2) functions just the same as an embedding vector of any length. Finding the distance between two data points in a standard 2 dimensional plane functions the same as finding the same distance in a 243-dimensional space, just with some extra calculations for the additional dimensions. As a result, these clustering algorithms will function the same in any n-dimensional space and can be thought of (for simplicity of visualization and thinking about the algorithms) as working in a 2 or 3 dimensional space. For the examples that accompany each algorithm, embeddings from the Universal Sentence Encoder are used for consistency and to give an example of how effectively each algorithm works with information rich, context encoded embeddings. Where possible, the algorithm parameters have been tuned to provide the best results possible with the data used. For each case, the clustering was applied only to the first 100 sentences to keep the memory usage reasonable. Because the original text data at this point has been transformed to a series of embeddings, we're well enough removed from the characteristics of the texts that clustering techniques should be chosen based on how well they capture the desired groups. Some of the algorithms below operate under a set of assumptions (convexity of clusters, uniform density, etc), but as with many things in the world of statistics sometimes violating assumptions doesn't actually cause a problem. Theory is theory, practice is practice!

### Affinity Propagation
Affinity propagation solves the problem of identifying similar datapoints by evaluating how good of an exemplar a given data point would be for any other data point. Affinity propagation does not require the number of clusters to be specified, and instead extracts the optimal number of clusters based on the data provided. The algorithm stores this information in two matrices:
- The *responsibility* matrix R<sub>*i,j*</sub> measures how good of an exemplar *j* is for *i* relative to all other *j*
- The *availability* matrix A<sub>*i,j*</sub> measures how appropriate it would be for *i* to choose *j* as its exemplar, taking into account the extent to which other *i* support *j* as their exemplar
I personally found the relationships between these two matrices very difficult conceptually, so I'll try and lay them out a bit less math-ily here. The responsibility matrix measures how well a given point represents another point, i.e. **"how similar/close are these two points?"**, while the availability matrix tells us how good of an exemplar each point would be by considering the number of surrounding points which could be represented best by that point, i.e. **"how good of an exemplar would this point be in the grand scheme of things?"**. R shows us which points are similar, A shows us which points should be exemplars
- **Steps**
	+ R and A are initialized as all-zero matrices
	+ Each value *i,j* of R is updated via a similarity function *s*, defaulting to the negative squared euclidean distance between *i* and *j*. This value is stored as *s(i,j)* - max(A(*i,j´*) + s(*i,j´*))
	+ Each value *i,j* of A is updated and stored as min(0,*r(k,k)* + sum(max(0,*r(i´,k)*)))
	+ These two steps are repeated until cluster boundaries stabilize for a set number of iterations
	+ Exemplars will be those points that have a positive sum of their R and A values
- **Example**

### Agglomerative
Agglomerative clustering is an implementation of hierarchical clustering techniques, which take a bottom up approach to clustering. There are a couple different distance metrics and linkage strategies which can be used in the calculations, but the general procedure considers each datapoint as its own cluster and merges similar clusters until a target n_clusters or distance threshold is reached
- **Steps**
	+ The initial cluster space is constructed with each datapoint considered as its own cluster
	+ Point wise distances are calculated with the provided metric
	+ Clusters are merged in accordance with the selected linkage strategy
	+ The previous 2 steps are repeated until the requisite n_clusters or distance threshold is satisfied
- **Parameters**
	+ *Affinity* is the distance metric used in linkage calculations and can take a range of common metrics
	+ *Linkage* is the minimization strategy for cluster merging, each linkage strategy prioritizes a different metric when combining clusters
		* ward - minimizes variance from cluster centroids
		* complete - minimizes the maximum distance between all cluster members
		* average - minimizes average distance between all cluster members
		* single - minimizes the minimum distance between all cluster members
- **Example**

### Birch
Birch (balanced iterative reducing and clustering using hierarchies) clustering is another hierarchical clustering method, and works by constructing a clustering feature (CF) tree, composed of CF entries, the shape and characteristics of which can be controlled via the branching factor and threshold parameters
- **Steps**
	+ The CF tree is constructed such that:
		* Each CF entry contains sub-clusters numbering fewer than the branching factor
		* Any subcluster with a radius larger than the threshold value has been split into multiple clusters as necessary
	+ The CF tree is scanned in order to remove outliers and group densely packed subclusters into larger ones, parameters permitting (*this step is sometimes omitted*)
	+ An agglomerative clustering algorithm is applied to the CF entries
- **Parameters**
	+ *Branching factor* defines the maximum number of subclusters which can exist in each CF entry before being split into multiple CF entries
- **Example**

### DBSCAN & HDBSCAN
DBSCAN and HDBSCAN both implement a density based clustering approach, with HDBSCAN extending the basic DBSCAN approach with a hierarchical framework. Unlike other clustering algorithms which take distance metrics or n_clusters as parameters, density based clustering allows specification of *what should constitute a cluster* by defining the minimum number of neighbors that should exist in a certain maximum distance from a given core sample datapoint. The combination of the min_samples, metric, and distance (epsilon) parameters creates the "density" used for cluster identification
- **Steps**
	+ A core sample which meets the neighborhood criterion as defined by min_samples and eps is identified
	+ All other core samples in the neighborhood of the original core sample are found
	+ Previous step is repeated for all newly identified core samples as necessary until all core samples in the cluster are identified
	+ All previous steps are repeated for any core samples not captured
	+ Any datapoints not encompassed after all core samples have been clustered are considered outliers or "noise"
	+ HDBSCAN then iterates over clustered data to identify clusters which can be merged while minimizing loss of cluster members within provided parameters
	+ HDBSCAN constructs a cluster hierarchy tree and select a height from which to pull clusters
- **Parameters**
	+ *Epsilon* defines how close 2 points must be in order to be considered "neighbors", not to be confused with a cluster radius as used in other algorithms
	+ *Metric* is the method to use when calculating distances between points
	+ *Minimum Samples* defines the number of neighbors a datapoint must have in order to be considered an exemplar (also called a core point in DBSCAN documentation)
	+ *Minimum Cluster Size* provides the smallest number of members a cluster can have. Member attrition exceeding this value will constitute a new cluster, and otherwise will be considered noise
- **Example**

### KMeans & Mini Batch KMeans
KMeans is perhaps the most ubiquitous clustering algorithm in use today. It's fairly lightweight and intuitive, and has a number of different parameters which allow for fine tuning on a task by task basis. Mini Batch KMeans uses the same basic framework as the base KMeans algorithm, but often achieves convergence faster and with less computing power by using a random subset of the data for each centroid initialization
- **Steps**
	+ A provided number of cluster centroids are initialized, sometimes chosen from real datapoints and sometimes chosen as random points
	+ Each datapoint is assigned to the nearest centroid
	+ Average coordinates for each previously defined centroid are calculated and compared to the randomly initialized centroid for each cluster
	+ Previous 3 steps are repeated until the distance change between iterations falls below a defined threshold
- **Example**

### Mean Shift
Mean shift clustering uses datapoint density to identify clusters, much like DBSCAN and HDBSCAN. The main differentiating feature of the mean shift algorithm is its use of density kernels in assessing clusters. Over the course of the algorithm, the location of each datapoint is iteratively *shifted* toward the *mean* location of neighboring points, which eventually converge on the local cluster centroid
- **Steps**
	+ Each datapoint is assigned a density kernel, usually a gaussian kernel. The width of this kernel can be specified, but is usually estimated automatically within the algorithm
	+ The kernel functions are summed across all datapoints in order to create a density function for the entire vector space
	+ Each point is moved towards the nearest density maximum until convergence, at which point it is assigned to its cluster centroid
- **Example**

### OPTICS
The OPTICS (Ordering Points To Identify Cluster Structure) algorithm is very similar to DBSCAN in that it begins from areas of high density and builds clusters outwards. OPTICS differs from other density based clustering algorithms in its use of "reachability", which considers the core distance (minimum distance required to classify a point as a core point) and the euclidean distance between a core sample and any given point. Reachability is set to equal the larger of these two values, and from the reachability scores a dendrogram is constructed, which is then used to identify clusters in the data
- **Steps**
	+ Core and euclidean distances are calculated for core samples in accordance with DBSCAN, selecting the reachability distance from these values
	+ All datapoints are arranged such that points closest in the vector space are neighbors in the ordering
	+ Cluster hierarchy and membership are assigned based on the reachability plot, allowing for varying cluster densities and noise thresholds
- **Parameters**
	+ *Metric* is the method to use when calculating distances between points
	+ *Minimum Samples* defines the number of neighbors a datapoint must have in order to be considered an exemplar (also called a core point in DBSCAN documentation)
	+ *Minimum Cluster Size* provides the smallest number of members a cluster can have. Member attrition exceeding this value will constitute a new cluster, and otherwise will be considered noise
- **Example**

### Spectral
Spectral clustering relies on a graph based representation of the data in order to parse clusters. The algorithm represents the datapoints as an adjacency matrix and assigns their similarity as edge weights. By doing so, clusters are identified by collections of edges with higher weights, with relatively low edge weights denoting likely clause boundaries
- **Steps**
	+ Adjacency matrix is constructed by computing similarity between points using the provided affinity metric
	+ Datapoints are projected onto a lower dimensional space via the Graph Laplacian Matrix
	+ A more basic clustering algorithm (usually KMeans) is applied to the projected data
- **Parameters**
	+ *Affinity* is the metric used to construct the affinity matrix, and can be thought of as an alternative to a traditional distance metric
- **Example**

## Dimensionality Reduction
Dimensionality reduction is a fairly broad range of techniques used to project a series of datapoints in a high dimensional space into a lower dimensional space. In this way, clustering and by extension most of what's been discussed thus far would be considered a type of dimensionality reduction. That being said, these dimensionality reduction techniques are not necessary to understand or implement the sentence embedding and clustering approaches above. Rather, the algorithms discussed here are primarily for the purpose of visualization. All of the techniques used here will be projecting sentence embeddings into 2 and 3 dimensional vector spaces, though for different tasks the number of dimensions can be specified as needed. For provided examples, the dimensionality reduction algorithm results are visualized with 2 and 3 dimensional plots, and for consistency the embeddings from Universal Sentence Encodings and the clusters from HDBSCAN are used. Because the original text data at this point has been transformed to a series of embeddings and groupings, we're well enough removed from the characteristics of the texts that dimensionality reduction techniques should be chosen based on effectiveness on a task by task basis

### Uniform Manifold Approximation & Projection
The UMAP algorithm is one of the more recent additions to the world of dimensionality reduction, as well as one of the more conceptually complex. The algorithm works by first constructing a topological representation of the data in its higher dimensional space. From here, a lower dimensional representation is created in order to mirror as closely as possible the higher dimensional topography. The optimization stage uses techniques common in machine learning spaces, including stochastic gradient descent
- **Parameters**
	+ *N Neighbors* defines how small of a scale UMAP will consider clusters on, leading to finer differentiation among nearby points at the potential drawback of extremely localized clusters
	+ *Minimum Distance* defines how close points must be before they can be considered as occupying the same space as one another. In practice, this also controls how much fine grained detail the constructed topology will capture
	+ *Metric* is the metric used by UMAP to calculate the distances between points. UMAP includes a range of supported metrics for a range of data types, including those for normalized and binary data
- **Example**

### Truncated Singular Value Decomposition
Truncated singular value decomposition (TSVD) is a constrained implementation of SVD, where the algorithm only calculates the number of singular values provided by the user. Because this technique will also be utilized by other algorithms and encompasses a lot of central ideas in dimensionality reduction, I'll go into more detail on this algorithm than the others
- **SVD**
	+ SVD is based on the idea of linear transformations. These transformations include vertical/horizontal scaling and rotation along an axis in a given vector space. Think of dragging the corners of a clipart image and you've got the idea
	+ Any linear transformation can be accomplished with one rotation, one scaling, and another rotation, and these operations can themselves be represented as matrices providing the constants used by the transformations
	+ SVD represents these transformations as matrix A, which is equal to matrices *UΣV* (rotation, scaling, rotation)
	+ SVD when used for dimensionality reduction identifies opportunities to represent the data in fewer dimensions, via a combination of 1) identifying elements from the transformation matrices which can be approximated to 0, or 2) representing the full matrices as some product or sum of 2 or more vectors, thereby allowing fewer dimensions to represent the same or *nearly* the same data
	+ This can be performed until a desired threshold of data similarity is reached or until a target number of dimensions is reached (in the case of TSVD)
- **Example**

### Principal Component Analysis & Variants (Incremental, Kernel, Sparse, Mini Batch Sparse)
Principal component analysis (PCA) is one of the most common dimensionality reduction techniques in use today. At its core, the algorithm takes higher dimensional data and calculates the covariance matrix for all of the dimensions. From there, it selects the desired number of dimensions which capture the most information about the data (have the highest covariance), and projects the data into this n dimensional space. Basic PCA has a couple variants which may be appropriate for certain tasks or datasets:
- **Incremental PCA**
	+ IPCA focuses on computational efficiency by processing the data in batches and only keeping the most important dimensions from batch to batch
- **Kernel PCA**
	+ KPCA leverages a range of kernels in order to enable non-linear dimensionality reduction, which can be useful for datasets which aren't linearly separable
- **Sparse PCA**
	+ SPCA brings more interpretability to standard PCA by trading the usually dense representation of the components for sparse representations, which allows for the mapping of components to original features more readily
- **Mini Batch Sparse PCA**
	+ MBSPCA applies the same approach as SPCA but speeds up computations in the same way that IPCA does, by processing the data in smaller, easier to handle batches
- **Example**
*Sparse and Mini Batch Sparse PCA examples are omitted as they require sparse embeddings, USE produces a dense embedding matrix*

### Gaussian & Sparse Random Projection
Random projection in the context of dimensionality reduction is based on the Johnson-Lindenstrauss lemma, which states that when dealing with high dimensional data, projecting the datapoints in random directions preserves the pairwise distances between the datapoints. The algorithm allows for random projection within a provided error range, to minimize/control the amount of data shift during the dimensionality reduction process. This holds true for both main variations on the algorithm, with the main difference in how the directions of the projection vectors are chosen
- **Gaussian Random Projection**
	+ Projection matrix elements are chosen from a gaussian distribution with μ=0
- **Sparse Random Projection**
	+ Projection matrix elements are chosen from a set of 0 and the positive and negative values of a user provided constant. Implementations of this approach often include a density parameter, which selects the aforementioned constant based on the number of dimensions in the original data
- **Example**

### t-Distributed Stochastic Neighbor Embedding
t-SNE utilizes the properties of t-distributions and joint probabilities to represent higher dimensional data in a lower dimensional vector space. The algorithm builds pairwise probability distributions for points in the original data such that similar points will have higher probabilities. The algorithm then compares these distributions to one constructed in the target dimensional space, and seeks to minimize variance between the two
- **Parameters**
	+ *Metric* specifies the distance/affinity metric to be used in the calculations, the choice of which can have a major impact on the results
- **Example**

### Factor Analysis
Factor analysis seeks to create a set of formulas which define how our observed data *x* might be generated from 1 or more latent variables. This generative model includes a user provided number of latent variables, which are associated with the lower dimensional space onto which the data is projected
- **Example**

### Independent Component Analysis
ICA is most commonly used in continuous signal data such as sound or image signals, but can be used for dimensionality reduction in some cases. The algorithm starts from the original data and assumes it is a mixture of component signals, each with a different contribution to the observed value. The main data (or signal) is then decomposed into a specified number of component signals in a way that maximizes the amount of information contributed by and the independence between each component
- **Example**

### Latent Dirichlet Allocation
LDA is the only dimensionality reduction technique here which is specifically tailored to text data, though it relies on similar conceptual and mathematical underpinnings as other approaches. It views the data (*corpus*) as a collection of *documents*, each probabilistically composed of a number of *topics* according to 1 or more latent variables. The algorithm then uses the topics and their associated words to infer the underlying latent variables that describe how topics are distributed in the corpus using joint probabilities (a la Bayes). For the below example, the USE embeddings had to be manipulated a bit because LDA doesn't play nice with negative values. Take the results with a grain of salt
- **Example**

### Nonnegative Matrix Factorization
As the name suggests, NMF assumes the data and all underlying components are non-negative. The algorithm factorizes the original data into vertical and horizontal components, and proceeds by minimizing the variance between the original data and the factorized components using a specified distance metric. For the below example, the USE embeddings had to be manipulated a bit because NMF doesn't play nice with negative values. Take the results with a grain of salt
- **Example**

#
