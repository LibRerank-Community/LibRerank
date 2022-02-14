# Paper List


## Overview
* [Accuracy-oriented Re-ranking](#accuracy-oriented-re-ranking)
    * [Learning by observed signals](#learning-by-observed-signals)
    * [Learning by counterfactual signals](#learning-by-counterfactual-signals)
* [Diversity-aware Re-ranking](#diversity-aware-re-ranking)
    * [Implicit Approaches](#implicit-approaches)
    * [Explicit Approaches](#explicit-approaches)
* [Fairness-aware Re-ranking](#fairness-aware-re-ranking)

## Accuracy-oriented Re-ranking
### Learning by observed signals

* __Learning a deep listwise context model for ranking refinement__，(SIGIR 2018), _Ai, Qingyao, Keping Bi, Jiafeng Guo, and W. Bruce Croft_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3209978.3209985) 

* __Globally Optimized Mutual Influence Aware Ranking in E-Commerce Search__, (IJCAI 2018), _Zhuang, Tao, Wenwu Ou, and Zhirong Wan_. [[pdf]](https://www.ijcai.org/proceedings/2018/0518.pdf) 

* __Seq2slate: Re-ranking and slate optimization with rnns__, (ArXiv 2018), _Bello, Irwan, Sayali Kulkarni, Sagar Jain, Craig Boutilier, Ed Chi, Elad Eban, Xiyang Luo, Alan Mackey, and Ofer Meshi_. [[pdf]](https://arxiv.org/pdf/1810.02019.pdf) 

* __Beyond Greedy Ranking: Slate Optimization via List-CVAE__, (ICLR 2018), _Jiang, Ray, Sven Gowal, Yuqiu Qian, Timothy Mann, and Danilo J. Rezende_. [[pdf]](https://arxiv.org/pdf/1803.01682.pdf) 

* __Personalised reranking of paper recommendations using paper content and user behavior__, (TOIS 2019), _Li, Xinyi, Yifan Chen, Benjamin Pettit, and Maarten De Rijke_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3312528?casa_token=8TYxtgF2MP0AAAAA:_PV20nMRTk6gSiQVC70AFWliFkYFK_Yrk0z--VZ3JC_CzqbwzHTJLVFeAagKWvvzVpMuhlY-2CxcFA) 

* __Personalized re-ranking for recommendation__, (RecSys 2019), _Pei, Changhua, Yi Zhang, Yongfeng Zhang, Fei Sun, Xiao Lin, Hanxiao Sun, Jian Wu et al_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3298689.3347000?casa_token=YLgQvHaEfpUAAAAA:f_Z-mefy-ws_s0Vg_mZu-g6IEAA2I-VcsNFg56PPDf-jbU0Pl2WbaY7galB0N73WqjrqTr4KmZkUcw) 

* __Personalized Flight Itinerary Ranking at Fliggy__, (CIKM 2020), _Huang, Jinhong, Yang Li, Shan Sun, Bufeng Zhang, and Jin Huang_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3340531.3412735?casa_token=N25_yCSF44kAAAAA:xC6XqmfkcQorZgNnip8hgDaUYFTOp_VOMKBgTSLXLjueFVu3kaax9m1LGCO9ZVgluOdf0To5CxLf7Q) 

* __Personalized Re-ranking with Item Relationships for E-commerce__, (CIKM 2020), _Liu, Weiwen, Qing Liu, Ruiming Tang, Junyang Chen, Xiuqiang He, and Pheng Ann Heng_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3340531.3412332?casa_token=yHkSC9M-bpEAAAAA:_knfgZwDd9ZthI6k75nmqWLgjya6GEwHyvr2UoNSvejM2FItqYBk9JhOc8IWcyoojrgXZW69n2LdTQ) 


* __Variation Control and Evaluation for Generative Slate Recommendations__, (WWW 2021), _Liu, Shuchang, Fei Sun, Yingqiang Ge, Changhua Pei, and Yongfeng Zhang_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3442381.3449864?casa_token=HDI6HLJpIX4AAAAA:mi0c7XsJb8uFz7QKCB6RGfB7mFM9D58UibMpj5FkuPqz6dzG3HIh4BpQTdVC9QJoUJsPx7CuiSXocw) 

* __Attention over Self-attention: Intention-aware Re-ranking with Dynamic Transformer Encoders for Recommendation__, (ArXiv 2022), _Lin, Zhuoyi, Sheng Zang, Rundong Wang, Zhu Sun, Chi Xu, and Chee-Keong Kwoh_. [[pdf]](https://arxiv.org/pdf/2201.05333.pdf) 

### Learning by counterfactual signals


* __Sequential evaluation and generation framework for combinatorial recommender system__, (ArXiv 2019), _Wang, Fan, Xiaomin Fang, Lihang Liu, Yaxue Chen, Jiucheng Tao, Zhiming Peng, Cihang Jin, and Hao Tian_. [[pdf]](https://arxiv.org/pdf/1902.00245.pdf?ref=https://githubhelp.com) 

* __Learning groupwise multivariate scoring functions using deep neural networks__, (ICTIR 2019), _Ai, Qingyao, Xuanhui Wang, Sebastian Bruch, Nadav Golbandi, Michael Bendersky, and Marc Najork_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3341981.3344218) 

* __Co-Displayed Items Aware List Recommendation__, (IEEE Access 2020), _Song, Junshuai, Zhao Li, Chang Zhou, Jinze Bai, Zhenpeng Li, Jian Li, and Jun Gao_. [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9052461) 

* __Setrank: Learning a permutation-invariant ranking model for information retrieval__, (SIGIR 2020), _Pang, Liang, Jun Xu, Qingyao Ai, Yanyan Lan, Xueqi Cheng, and Jirong Wen_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3397271.3401104?casa_token=MZQmPBqZ-_0AAAAA:Ynip1qaF0b4fcSinCbfcJllBmpmIwjlspqzULUIIS2wXWFYl6l1w2fA31SXAol1-jZr6fPpO8TN7vw) 

* __Generator and Critic: A Deep Reinforcement Learning Approach for Slate Re-ranking in E-commerce__, (ArXiv 2020), _Wei, Jianxiong, Anxiang Zeng, Yueqiu Wu, Peng Guo, Qingsong Hua, and Qingpeng Cai_. [[pdf]](https://arxiv.org/pdf/2005.12206.pdf) 

* __Revisit Recommender System in the Permutation Prospective__, (ArXiv 2021), _Feng, Yufei, Yu Gong, Fei Sun, Junfeng Ge, and Wenwu Ou_. [[pdf]](https://arxiv.org/pdf/2102.12057.pdf) 

* __GRN: Generative Rerank Network for Context-wise Recommendation__, (ArXiv 2021), _Feng, Yufei, Binbin Hu, Yu Gong, Fei Sun, Qingwen Liu, and Wenwu Ou_. [[pdf]](https://arxiv.org/pdf/2104.00860.pdf) 


* __Context-aware Reranking with Utility Maximization for Recommendation__, (ArXiv 2021), _Xi, Yunjia, Weiwen Liu, Xinyi Dai, Ruiming Tang, Weinan Zhang, Qing Liu, Xiuqiang He, and Yong Yu_. [[pdf]](https://arxiv.org/pdf/2110.09059.pdf) 

* __AliExpress Learning-To-Rank: Maximizing Online Model Performance without Going Online__, (TDKE 2021), _Huzhang, Guangda, Zhenjia Pang, Yongqing Gao, Yawen Liu, Weijie Shen, Wen-Ji Zhou, Qing Da et al_. [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9495161&casa_token=phJ-uT1T8TcAAAAA:8bSWyJtWjs9f55MIsUvmpM_cdTDR7aRg-gnG1TGf-II428XrkcN9NN6CszedtCtpkwkO_wgQyg&tag=1) 





## Diversity-aware Re-ranking

### Implicit Approaches

* __Modeling document novelty with neural tensor network for search result diversification__, (SIGIR 2016), _Xia, Long, Jun Xu, Yanyan Lan, Jiafeng Guo, and Xueqi Cheng_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/2911451.2911498?casa_token=1ER68Z_VUt8AAAAA:CoHIiuLTrXlDxHXB0Cwdlxz6VtJEpH4Iv7s2ASQ6g0n1WYsnnFz48m09wluLbBoAFeLw6E4zpcfx3g) 

* __Adapting Markov decision process for search result diversification__, (SIGIR 2017, _Xia, Long, Jun Xu, Yanyan Lan, Jiafeng Guo, Wei Zeng, and Xueqi Cheng_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3077136.3080775?casa_token=D7p6_EICDocAAAAA:ySUupBBu2jcLjOiQAPfmNcE5eM9VfOX8ZJXLTI_RvM3ndDogAEG9A5KaAjGYv-L5WcvqjF5no8xc1g) 

* __From greedy selection to exploratory decision-making: Diverse ranking with policy-value networks__, (SIGIR 2018), _Feng, Yue, Jun Xu, Yanyan Lan, Jiafeng Guo, Wei Zeng, and Xueqi Cheng_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3209978.3209979) 

* __Diversification-aware learning to rank using distributed representation__, (WWW 2021), _Yan, Le, Zhen Qin, Rama Kumar Pasumarthi, Xuanhui Wang, and Michael Bendersky_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3442381.3449831?casa_token=AmDHmrD0SuIAAAAA:J17u8EChXpVYpkm7JOcd71l15B2LlJQnzIY5DduV0GOh4btiHnAwytp1tqQBx_5IPWzoKmbi4ekwNA) 

### Explicit Approaches

* __Learning to diversify search results via subtopic attention__, (SIGIR 2017), _Jiang, Zhengbao, Ji-Rong Wen, Zhicheng Dou, Wayne Xin Zhao, Jian-Yun Nie, and Ming Yue_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3077136.3080805?casa_token=dq9HBxnzzkoAAAAA:4PvJ2d9U6lvIW2sLWdBHMcOazy-k1_e_bMHzHm0YN7EgDBk0IT5EGIFWWjfx8Ifojs8SOaiSch2FLA) 


* __DVGAN: A minimax game for search result diversification combining explicit and implicit features__, (SIGIR 2020), _Liu, Jiongnan, Zhicheng Dou, Xiaojie Wang, Shuqi Lu, and Ji-Rong Wen_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3397271.3401084?casa_token=tVKJs-6Fc8QAAAAA:Z98BPi8rWTPp8_Ton_JX969bw5ZhQd4o6wXvmO3aSc4uAacosQs-N4OsWvgj9KrkmU7UWCnNiStTMw) 

* __Diversifying search results using self-attention network__, (CIKM 2020), _Qin, Xubo, Zhicheng Dou, and Ji-Rong Wen_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3340531.3411914?casa_token=lpyMjM-ZjaYAAAAA:3n2qeg_edGNimO84OXgHA4ya3R1MxqwOxYE1tnX5xlDL7hlKDawYtXbM8XXfbA4Ks_qstAs5KMImVQ) 

* __Managing diversity in Airbnb search__, (KDD 2020), _Abdool, Mustafa, Malay Haldar, Prashant Ramanathan, Tyler Sax, Lanbo Zhang, Aamir Manaswala, Lynn Yang, Bradley Turnbull, Qing Zhang, and Thomas Legrand_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403345?casa_token=-pdIr0hXuEkAAAAA:KhXY8JME8_FYIXNUFZFWtrEmvD_QiAMpYvltwJ9aXMGJXTNevh1DjYRyj7ZigFiaOph8kIhb8egwCQ) 

## Fairness-aware Re-ranking


* __Using image fairness representations in diversity-based re-ranking for recommendations__, (UMAP 2018), _Karako, Chen, and Putra Manggala_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3213586.3226206?casa_token=zwlfoyDpOs0AAAAA:umTNePRI3Pmb5gEgulHg5OSa3ewnIU9PCduxiGSqS2rMxOijBXRKnR7lm0rXW-8crr70iMJYFWqkvw) 

* __Policy learning for fairness in ranking__, (NeuIPS 2019), _Singh, Ashudeep, and Thorsten Joachims_. [[pdf]](https://proceedings.neurips.cc/paper/2019/file/9e82757e9a1c12cb710ad680db11f6f1-Paper.pdf) 

* __Computationally Efficient Optimization of Plackett-Luce Ranking Models for Relevance and Fairness__, (SIGIR 2021), _Oosterhuis, Harrie_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3404835.3462830) 

* __Policy-gradient training of fair and unbiased ranking functions__, (SIGIR 2021), _Yadav, Himank, Zhengxiao Du, and Thorsten Joachims._. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3404835.3462953?casa_token=kYWENGXPn2wAAAAA:rU1HQPzZSr7K7u_vkmS1Z7hJLAJA3tNH6qJDzeoomBivUQeHHx-dsjrT943ruovRvCbu29Q4DYR6uw) 

* __Fairness among New Items in Cold Start Recommender Systems__, (SIGIR 2021), _Zhu, Ziwei, Jingu Kim, Trung Nguyen, Aish Fenton, and James Caverlee_. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3404835.3462948?casa_token=j5_fJQeOkLgAAAAA:oTx_4VpeJ0ICNiy5UDBc2QD0YrJjzal21-aM6RmzoiuBSaA51IQ3oWeyRJnMfAs3eqvcOOYF0s3YQA) 
