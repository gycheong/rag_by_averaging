# Improving RAG by Averaging
This repository contains a capstone project for the [Erdős Institute Data Science Boot Camp](https://www.erdosinstitute.org/). The data used in this notebook is provided by Jason Morgan at AwareHQ.

**Presentation slides**: [slides.pdf](https://github.com/gycheong/rag_by_averaging/blob/main/slides.pdf)

**Remark**. We note that the evaluation numerics in slides are different as they were tested with different random number generators. However, the results in improvement we aim are consistent.

**Main notebook file**: [RAG.ipynb](https://github.com/gycheong/rag_by_averaging/blob/main/RAG.ipynb)

**Files needed to reproduce the codes**:
* [Reddit data](https://drive.google.com/file/d/1Xc-GCpAQFGfTUOHOxwBsCNFFB9klgF5A/view?usp=sharing): put it in the same folder as the notebook file
* SBERT data for the sentence vectors: put it in the same folder as the notebook file
  1. [WarlmartEmployees_embeddings.pt](https://drive.google.com/file/d/1oKnpsSeCVqx4Ougzyn5KotKkA-c0TGil/view?usp=drive_link)
  2. [TalesFromYourBank_embeddings.pt](https://drive.google.com/file/d/1iJruY2m8i9aoLhZJh2j-COBMAONC2JTa/view?usp=drive_link)
  3. [RiteAid_embeddings.pt](https://drive.google.com/file/d/1st8jUyOxagBouyzZkSrizuWPR52RkYGG/view?usp=drive_link)
  4. [PaneraEmployees_embeddings.pt](https://drive.google.com/file/d/1ZaBtGKBq-OixqTPI0bfJLIzQh6Hm5vNe/view?usp=drive_link)
  5. [KrakenSupport_embeddings.pt](https://drive.google.com/file/d/1X0o_ViY5T7nOYoKo-i9Hdx3i2SQpPDdo/view?usp=drive_link)
  6. [FedEmployees_embeddings.pt](https://drive.google.com/file/d/1_dbP_NdyrmsF4o4HkzyU3KQjeDiRV25p/view?usp=drive_link)
  7. [Chase_embeddings.pt](https://drive.google.com/file/d/1mJWw2Qk4Oni1Ikvyshxsd0hZhfWSpID5/view?usp=drive_link)
  8. [cabincrewcareers_embeddings.pt](https://drive.google.com/file/d/1nygxNEg22uiCwz4ZTeFSVebJEEfVYS5B/view?usp=drive_link)
  9. [BestBuyWorkers_embeddings.pt](https://drive.google.com/file/d/1ZpnV7dXAleWoHUgnPZO_hp_FIN77EGVK/view?usp=drive_link)

## Authors
* [Gilyoung Cheong](https://www.linkedin.com/in/gycheong/)
* [Qidu Fu](https://www.linkedin.com/in/qidu-chidu-fu-212272236/)
* [Junichi Koganemaru](https://www.linkedin.com/in/junichi-koganemaru/)
* [Xinyuan Lai](https://www.linkedin.com/in/xinyuan-lai-98369810b/)
* [Sixuan Lou](https://www.linkedin.com/in/sixuan-lou/)
* [Dapeng Shang](https://www.linkedin.com/in/dapeng-shang-654316105/)

## Overview
We implement a specific pipline of [Retrieval-Augmented Generation (RAG)](https://aws.amazon.com/what-is/retrieval-augmented-generation/) using [SBERT](https://arxiv.org/abs/1908.10084) developed by Nils Reimers and Iryna Gurevych. Experimentally, we show that the one we implement (averaging RAG) is better than the other baseline one (naive RAG) in retrival based on two reasonable relative performance metrics. 

The documentation for the SBERT API for Python is available in [this link](https://sbert.net/). We use SBERT to find relevant comments to a query about employees from several subreddits. For LLM generation, we use Gemma 2B-IT using HuggingFace API, which we learned from [this article](https://huggingface.co/learn/cookbook/en/rag_with_hugging_face_gemma_mongodb) by Richmond Alake. The following diagram summarizes our process of **averaging RAG**, which we discuss more in detail below:
![image](https://github.com/gycheong/rag_by_averaging/assets/139825285/d8bada53-bd3a-4621-a77f-87db7d81fcc1)

## Data Processing

Prior to feeding the text data into SBERT for vectorization, we 
ﬁrst took some steps to clean the data to improve the quality of 
the vectorization and retrieved comments. This included:
* Removing  missing  text  data,  including  deleted  and  removed
comments.
* Removing  short  and  long  words.
* Stemming  and  lemmatizing  words  using  NLTK 
WordNetLemmatizer
* Removing  emoticons  and  emojis
* Spelling  out  common  chat  abbreviations  (e.g.   afaik  →  As  far 
as  I  know)

We  deleted  only  4000  comments  (roughly  4%)  in  our  100k  sample
database.

## Naive RAG vs Averaging RAG

* The **naive RAG** for us means that we find top 5 relevant comments to the query and use them to generate a response to the query using LLM.
* For the **averaging RAG**, we generate more similar queries to the original query using synthetic query generators and re-rank the comments by the average cosine similarity (i.e., averaging the cosine similaities of each comment to all the possible queries). Then we use the top 5 comments to generate a response to the query using LLM.

* **Clustering**: to speed up the runtime, we group the comments into clusters using K-means (K=4) clustering based on their vector representations with respect to cosine similarity. We stored the average vector representations of each cluster for quick comparison. This allowed us to cut the runtime down to under 1 second on average.

## Benefits of SBERT vs BERT

SBERT (Sentence Bert) is based on [BERT (Bidirectional Encoder Representations from Transformer)](https://arxiv.org/abs/1810.04805) developed by Google. From inspection, there are clear benefits of using SBERT over BERT for our purpose.

1. BERT is designed to generate vectors that correspond to individual words (or more precisely, *subwords*) to a sentence, so each sentence is converted into not just a vector but a sequence of vectors. Hence, in order to examine the similaritiy of two sentences, we need to either pick one word or take the average of the vectors, which did not yield satisfying results.

2. Because BERT converts every subword as a vector, in order to fully use it, we need to use a lot more storage. In an experiement, examining 10400 comments required 11.8GB with BERT while it only required 91.6MB with SBERT.

3. For BERT, the query and the comments (i.e., information to answer the query) need to be proceeded together when we embedd them as (sequences of) vectors. For SBERT, we can vectorize the comments first and then indepedently vectorize the query later.

## Query and LLM generate responses from top 5 comments
* Query: **How many PTOs does a regular employee have a year?**
* LLM Response with Naive RAG: **Regular employees typically earn 48 hours of paid time off per year, with the exception of a few states that have unlimited PTO by state law.**
* LLM Response with Averaging RAG: **According to the passage, regular employees have a maximum of 48 hours of PTO per year.**

**Remark**. In practice, it would make more sense to apply the above question on a specific subreddit instead of various subreddits. We have previously tried it for the Walmart employees subreddit and got a similar improvement in our results. Our current deployment is to ensure that we can handle larger data than just having one subreddit.

## Evaluation of retrieval

Note that it is rather difficult to say which LLM responses are better. Moreover, we note that our goal is NOT to get the answer that is absolutely correct but a relevant one among the reddit comments that we put in. For example, the answer may change over time, unless we update the input comments.

Hence, we use use both of the LLM responses as ground truths and compare the top 50 retrievals from the two methods:
* Method 1: Naive RAG using cosine similairties against the original query
* Method 2: Not-so-naive RAG using average cosine similairties against multiple similar queries, including the original one

### Evaluation method 1: Cosine Precision

Let $\boldsymbol{t}_1$ and $\boldsymbol{t}_2$ be the truth vectors. For each vector $\boldsymbol{v}$ from a batch, the cosine similarities $\cos(\boldsymbol{t}_1, \boldsymbol{v})$ and $\cos(\boldsymbol{t}_2, \boldsymbol{v})$ are in the interval $[-1, 1]$, but in all of our examples, we know they are in $[0, 1]$. We simply take the average of the two to measure how truthful $\boldsymbol{v}$ is. Note that the closer the average is to $1$, the more truthful $\boldsymbol{v}$ is.

Recall the definition of **precision**:
$$\mathrm{Precision} := \frac{\mathrm{Relevant \ retrieved \ instances}}{\mathrm{All \ retrieved \ instances}}.$$

Given a batch $B$, we define the **cosine precision** as follows:

$$\mathrm{Cosine \ Precision \ of } \ B := \frac{1}{2|B|}\sum_{\boldsymbol{v} \in B}  (\cos(\boldsymbol{t}_1, \boldsymbol{v}) + \cos(\boldsymbol{t}_2, \boldsymbol{v})).$$

Indeed, we do see an improvement in our method from the naive RAG from 0.86954928 to 0.8786808:

![image](https://github.com/gycheong/rag_by_averaging/assets/139825285/92f72a9d-68aa-4273-8e39-19cc13e157b1)



### Evaluation method 2: Ranked Cosine Precision

Assume we retrieved $K$ comments in the context, ranked as $B = (x_1, \ldots, x_K)$.

We call the **precision at rank $m$** the cosine precision for the truncated context $B_m := (x_1, \ldots, x_m)$. And the ranked cosine precision is the average of these precisions.

$$
\text{Ranked Cosine Precision of } B := \frac{1}{K} \sum_{m = 1}^{K} \text{Cosine Precision of } B_m.
$$

Under this measurement, those comments ranked higher in the retrieved context will have a higher impact to the precision. We also see an improvement in our method from the naive RAG from 0.87982534 to 0.88779141 in this metric as well:

![image](https://github.com/gycheong/rag_by_averaging/assets/139825285/6d0d65e2-1347-4591-a312-b2ab8c7f62a0)


## Conclusion and future directions

As we have seen in the example above, our averaging method improves the overall retrieval better by getting rid of possibly unrelated retrieved data by comparisions with multiple similar queries to the original one. The LLM API we are using took almost 30 seconds to generate responses, but it is evident that any stronger LLM we use would only make the process faster.
