# Improving RAG by Averaging
This notebook is written as a part of the cap stone project for the [Erdős Institute Data Science Boot Camp](https://www.erdosinstitute.org/). The data used in this notebook is provided by Jason Morgan at AwareHQ. We use Gemma 2B-IT using HuggingFace API, which we learned from [this article](https://huggingface.co/learn/cookbook/en/rag_with_hugging_face_gemma_mongodb) by Richmond Alake.

**Main notebook file**: [RAG.ipynb](https://github.com/gycheong/rag_by_averaging/blob/main/RAG.ipynb)

## Authors
* Gilyoung Cheong
* Qidu Fu
* Junichi Koganemaru
* Xinyuan Lai
* Sixuan Lou
* Dapeng Shang

## Overview
We implement two pipelines of [Retrieval-Augmented Generation (RAG)](https://aws.amazon.com/what-is/retrieval-augmented-generation/) using [SBERT](https://arxiv.org/abs/1908.10084) developed by Nils Reimers and Iryna Gurevych, and we show that the one we implement (not-so-naive RAG) is better than the other baseline one (naive RAG) in retrival. The documentation for the SBERT API for Python is available in [this link](https://sbert.net/). We use SBERT to find relevant comments to a query about Walmart employees from the [Walmart Employees subreddit](https://www.reddit.com/r/WalmartEmployees/); we only use 10400 comments from previously saved data.

## Naive RAG vs Not-so-naive RAG

* The naive RAG for us means that we find top 5 relevant comments to the query and use them to generate a response to the query using LLM.
* For the not-so-niave RAG, we let our LLM generate more similar queries to the original query and re-rank the comments by the average cosine similarity (i.e., averaging the cosine similaities of each comment to all the possible queries). Then we use the top 5 comments to generate a response to the query using LLM.

## Benefits of SBERT vs BERT

SBERT (Sentence Bert) is based on [BERT (Bidirectional Encoder Representations from Transformer)](https://arxiv.org/abs/1810.04805) developed by Google. From inspection, there are clear benefits of using SBERT over BERT for our purpose.

1. BERT is designed to generate vectors that correspond to individual words (or more precisely *subwords*) to a sentence, so each sentence is converted into not just a vector but a sequence of vectors. Hence, in order to examine the similaritiy of two sentences, we need to either pick one word or take the average of the vectors, which did not yield satisfying results.

2. Because BERT converts every subword as a vector, in order to fully use it, we need to use a lot more storage. For our purpose of examining 10400 comments, it required 11.8GB with BERT while it only required 91.6MB with SBERT.

3. For BERT, the query and the comments (i.e., information to answer the query) need to be proceeded together when we embedd them as (sequences of) vectors. For SBERT, we can vectorize the comments first and then indepedently vectorize the query later.

## Query and LLM generate responses from top 5 comments
* Query: **How many PTOs does a regular employee have a year?**
* LLM Response with Naive RAG: **Regular employees are entitled to 1 hour of paid time off per 30 hours worked, with a maximum of 48 hours per year.**
* LLM Response with Not-so-naive RAG: **An employee is entitled to 68 hours of paid time off per year.**

## Evaluation of retrieval

Note that it is rather difficult to say which LLM responses are better. Moreover, we note that our goal is NOT to get the answer that is absolutely correct but a relevant one among the reddit comments that we put in. For example, the answer may change over time, unless we update the input comments.

Hence, we use use both of the LLM responses as ground truths and compare the top 50 retrievals from the two methods:
* Method 1: Naive RAG using cosine similairties against the original query
* Method 2: Not-so-naive RAG using average cosine similairties against multiple similar queries, including the original one

### Evaluation method: Cosine Precision

Let $\boldsymbol{t}_1$ and $\boldsymbol{t}_2$ be the truth vectors. For each vector $\boldsymbol{v}$ from a batch, the cosine similarities $\cos(\boldsymbol{t}_1, \boldsymbol{v})$ and $\cos(\boldsymbol{t}_2, \boldsymbol{v})$ are in the interval $[-1, 1]$, but in all of our examples, we know they are in $[0, 1]$. We simply take the average of the two to measure how truthful $\boldsymbol{v}$ is. Note that the closer the average is to $1$, the more truthful $\boldsymbol{v}$ is.

Recall the definition of **precision**:
$$\mathrm{Precision} := \frac{\mathrm{Relevant \ retrieved \ instances}}{\mathrm{All \ retrieved \ instances}}.$$

Given a batch $B$, we define the **cosine precision** as follows:

$$\mathrm{Cosine \ Precision \ of } \ B := \frac{1}{2|B|}\sum_{\boldsymbol{v} \in B}  (\cos(\boldsymbol{t}_1, \boldsymbol{v}) + \cos(\boldsymbol{t}_2, \boldsymbol{v})).$$

Indeed, we do see an improvement in our method from the naive RAG from 0.83449691 to 0.85419629:
![image](https://github.com/gycheong/rag_by_averaging/assets/139825285/05d6f8d1-b841-45a8-92c8-fda35f7d6551)

## Conclusion and future directions

As we have seen in the example above, our averaging method improves the overall retrieval better by getting rid of possibly unrelated retrieved data by comparisions with multiple similar queries to the original one. The LLM API we are using took about one minute to generate 10 similar queries, and we could only use half of them for our purpose to assure the quality of our result. It is evident that any stronger LLM we use would not only make the process faster, but it would also generate more similar queries that would result in an even better retrieval outcome.
