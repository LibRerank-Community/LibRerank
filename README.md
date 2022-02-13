# LibRerank
LibRerank is a toolkit for re-ranking algorithms. There are a number of re-ranking algorithms, such as PRM, DLCM, GSF, miDNN, SetRank, EGRerank, Seq2Slate. It also supports LambdaMART and DNN as initial ranker.

## Get Started

#### Create virtual environment(optional)
```
pip install --user virtualenv
~/.local/bin/virtualenv -p python3 ./venv
source venv/bin/activate
```

#### Install LibRerank from source
```
git clone https://github.com/LibRerank-Community/LibRerank.git
cd LibRerank
make init 
```

#### Run example
Run initial ranker
```
bash example/run_ranker.sh
```
Run re-ranker
```
bash example/run_reranker.sh
```
We can choose to enter a config file like `example/run_reranker.sh` via the parameter `setting_path`. The config files for the different models can be found in `example/config`. We can also set various parameters directly from the command line. A list of supported parameters can be found in `run_init_ranker.py` and `run_reranker.py`.

## Structure

### Initial rankers
DNN: a naive algorithm that directly train a multi-layer perceptron network with input labels (e.g., clicks).

LambdaMART: the implementation of the LambdaMART model in <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf">*From RankNet to LambdaRank to LambdaMART: An Overview*</a>
### Re-ranking algorithms
DLCM: the implementation of the Deep Listwise Context Model in <a href="https://arxiv.org/pdf/1804.05936.pdf">*Learning a Deep Listwise Context Model for Ranking Refinement*</a>.

PRM: the implementation of the Personalized Re-ranking Model in <a href="https://arxiv.org/pdf/1904.06813.pdf">*Personalized Re-ranking for Recommendation*</a>

GSF: the implementation of the Groupwise Scoring Function in <a href="https://arxiv.org/pdf/1811.04415.pdf">*Learning Groupwise Multivariate Scoring Functions Using Deep Neural Networks*</a>.

miDNN: the implementation of the miDNN model in <a href="https://www.ijcai.org/proceedings/2018/0518.pdf">*Globally Optimized Mutual Influence Aware Ranking in E-Commerce Search*</a>

SetRank: the implementation of the SetRank model in <a href="https://arxiv.org/abs/1912.05891">*SetRank: Learning a Permutation-Invariant Ranking Model for Information Retrieval*</a>.

Seq2Slate: the implementation of sequence-to-sequence model for re-ranking in <a href="https://arxiv.org/pdf/1810.02019.pdf">*Seq2Slate: Re-ranking and Slate Optimization with RNNs*</a>

EGRerank: the implementation of the Evaluator-Generator Reranking in <a href="https://arxiv.org/pdf/2003.11941.pdf">*AliExpress Learning-To-Rank: Maximizing Online Model Performance without Going Online*</a>


### Data

We process two datasets, [Ad](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56) and [PRM Public](https://github.com/rank2rec/rerank), containing user and item features with recommendation lists for the experimentation with personalized re-ranking. The details of processed datasets are summarized in the following table

| Dataset    | #item     | #list     | # user feature | #  item feature |
| ---------- | --------- | --------- | -------------- | --------------- |
| Ad         | 349,404   | 483,049   | 8              | 6               |
| PRM Public | 2,851,766 | 1,295,496 | 3              | 24              |

Depending on the length of the initial ranking, the maximum length of initial lists (re-ranking size n) is set to 10 and 30 for Ad and PRM Public, respectively.
#### Ad

The original [Ad dataset](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56) records 1 million users and 26 million ad display/click logs, with 8 user profiles (e.g., id, age, and occupation), 6 item features (e.g., id, campaign, and brand). Following previous work, We transform records of each user into ranking lists according to the timestamp of the user browsing the advertisement. Items that have been interacted with within five minutes are sliced into a list and the processed data is avaliable [here](https://github.com/LibRerank-Community/LibRerank/tree/master/Data/ad). The detailed process is [here](https://github.com/LibRerank-Community/LibRerank/blob/master/Data/preprocess_ad.py).

#### PRM public

The original [PRM public dataset](https://github.com/rank2rec/rerank) contains re-ranking lists from a real-world e-commerce RS. Each record is a recommendation list consisting of 3 user profile features, 5 categorical, and 19 dense item features.  Due to the memory limitation, we randomly sample 10% of lists and remained data is avaliable [here](https://drive.google.com/drive/folders/1c8HPVFAsLP6BwDzDRjd2Xs-BVP117uWQ?usp=sharing). The detailed process is [here](https://github.com/LibRerank-Community/LibRerank/blob/master/Data/preprocess_prm.py).

