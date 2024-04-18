# Improving RAG by Averaging
This notebook is written as a part of the cap stone project for the [Erdős Institute Data Science Boot Camp](https://www.erdosinstitute.org/). The data used in this notebook is provided by Jason Morgan at AwareHQ. We use Gemma 2B-IT using HuggingFace API, which we learned from [this article](https://huggingface.co/learn/cookbook/en/rag_with_hugging_face_gemma_mongodb) by Richmond Alake.

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
