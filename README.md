## What is Finance-NLP?

Text 데이터는 금융 분야에서 시계열 데이터만큼 중요하게 사용될 수 있다. 주식을 구매하고자 할 때, 뉴스를 읽고서 결정하기 때문이다. 과거에는 Text 데이터들도 사람이 수동으로 관리할 수 있는 양과 변동성이었지만, 최근 성장으로 인해 더 이상 불가능하게 되었다. 최근 NLP의 발전은 이러한 문제를 해결할 수 있도록 도와주고 있고, 전문가들이 다양한 판단을 할 수 있도록 도움을 주고 있다. 현재 혹은 앞으로 Finance-NLP가 활용될 수 있는 분야는 다음과 같다.

1. Market Analysis
    - classification / clustering 기법을 활용하여 market을 분석할 때 사용 가능
    - Micro(Stock Price Prediction) / Macro(Market Movement)와 같은 방법으로 활용
2. Risk Management
    - classification과 같은 방법을 사용하여 사기 혹은 자금 세탁등을 탐지할 수 있음
3. Finance Sentiment Analysis
    - 일반적인 Sentiment Analysis와는 다르게 금융 Sentiment Analysis는 시장이 뉴스에 어떤 반응을 보일지, 주가가 하락할지 상승할지 등을 살펴보는 것이 목적이다.
        - CEO가 사임했다는 뉴스는 보통 부정적인 감정을 가질 확률이 높고, 주가에 부정적인 영향을 미칠 것이다.
        - 하지만 CEO가 실적이 좋지 않았다면, 사임 소식은 긍정적인 영향을 줄 것
4. Asset or Portfolio Management
    - 비정형 문서를 NLP로 분석하여 자산 및 포트폴리오 선택을 최적화할 수 있음
        - internal: 내부 문서를 활용
        - external: 트위터와 같은 외부 문서 활용
5. Customer Engagement
    - Question & Answering, Dialog, Chatbot과 같이 고객과 상호작용하는 분야

## Finance-NLP Papers

### Market Analysis
- (EMNLP 2020) [Deep Attentive Learning for Stock Movement Prediction From Social Media Text and Company Correlations](https://aclanthology.org/2020.emnlp-main.676/)
- (EACL 2021) [FAST: Financial News and Tweet Based Time Aware Network for Stock Trading](https://aclanthology.org/2021.eacl-main.185/)
- (ACL 2022) [Incorporating Stock Market Signals for Twitter Stance Detection](https://aclanthology.org/2022.acl-long.281.pdf)
    - M&A와 관련 있는 회사들(구매자, 피구매자)의 주가 Trend를 Tweet 데이터를 통해 예측
- (ACL 2022) [Guided Attention Multimodal Multitask Financial Forecasting with Inter-Company Relationships and Global and Local News](https://aclanthology.org/2022.acl-long.437.pdf)

### Risk Management
- (ALTA 2015) [Domain Adaption of Named Entity Recognition to Support Credit Risk Assessment](https://aclanthology.org/U15-1010.pdf)
    - Finance NER

### Sentiment Analysis
- (IJCNLP 2015) Nguyen, T. H., & Shirai, K. [Topic modeling based sentiment analysis on social media for stock market prediction](https://aclanthology.org/P15-1131.pdf)
- (ACL-ECONLP 2018) Tobias Daudert, Paul Buitelaar and Sapna Negi. [Leveraging News Sentiment to Improve Microblog Sentiment Classification in the Financial Domain](https://www.aclweb.org/anthology/W18-3107/).
- Dogu Araci, [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/abs/1908.10063), arxiv.org(2019)

### Pre-trained Model(PLM)
- (ACL 2022) [Buy Tesla, Sell Ford: Assessing Implicit Stock Market Preference in Pre-trained Language Models](https://aclanthology.org/2022.acl-short.12.pdf)
    - 언어 모델은 streotype을 가지고 있고, 이는 FinBERT에서도 발견
        - 대부분 시장 상황을 긍정적으로 보고 있으며, 일부 종목들에 대해서는 완전 부정적으로 판단
    - 편향된 LM 모델로 추론을 진행하면 틀린 결론을 얻을 수 있음
        - positive가 많기 때문에, 부적절한 종목에 대해 buy라고 할 수 있음
    - 논문에서는 데이터셋을 잘 정제할 필요성에 대해 언급하고 있으며, 실험 결과를 보면 엄청 심각해 보이지는 않음
- (SIGIR 2022) [Structure and Semantics Preserving Document Representations](https://dl.acm.org/doi/10.1145/3477495.3532062)
    - document representation을 만드는 방법론
    - 기존 negative sampling으로 표현되는 triplet loss에 추가적으로 "구조적으로" 유사한가? 라는 loss를 추가하여 Quintuplet loss를 구축하여 학습
### Customer Engagement
- (EMNLP 2021) [FINQA: A Dataset of Numerical Reasoning over Financial Data](https://arxiv.org/pdf/2109.00122.pdf)
        - S&P 500 report에서 QA 데이터셋 구축 및 모델 연구
        - JP Morgan
- (ACL 2021) [TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance](https://arxiv.org/pdf/2105.07624.pdf)
### Forex(Exchange Rate)
- (IEEE Access) [BERTFOREX: Cascading Model for Forex Market Forecasting Using Fundamental and Technical Indicator Data Based on BERT](https://ieeexplore.ieee.org/document/9715051)
- (Soft Computing) [Foreign exchange currency rate prediction using a GRU-LSTM hybrid network](https://www.sciencedirect.com/science/article/pii/S2666222120300083)

### Relation Extraction
- (FinNLP2022) [No Stock is an Island: Learning Internal and Relational Attributes of Stocks with Contrastive Learning](https://mx.nthu.edu.tw/~chungchichen/FinNLP2022_IJCAI/1.pdf)

- (FinNLP2022) [How Can a Teacher Make Learning From Sparse Data Softer? Application to Business Relation Extraction](https://mx.nthu.edu.tw/~chungchichen/FinNLP2022_IJCAI/4.pdf)

### Fi-NER
- (ALTA 2015) [Domain Adaption of Named Entity Recognition to Support Credit Risk Assessment](https://aclanthology.org/U15-1010.pdf)
    - Finance NER
- (Springer 2021) [Extraction and Representation of Financial Entities from Text](https://www.researchgate.net/publication/352271164_Extraction_and_Representation_of_Financial_Entities_from_Text)
- (arxiv 2022) [FinBERT-MRC: financial named entity recognition using BERT under the machine reading comprehension paradigm](https://arxiv.org/abs/2205.15485)
    - 중국어 Dataset임
- (ACL 2022) [FiNER: Financial Numeric Entity Recognition for XBRL Tagging](https://aclanthology.org/2022.acl-long.303/)
- (FinNLP 2022) [AdaK-NER: An Adaptive Top-K Approach for Named Entity Recognition with Incomplete Annotations](https://mx.nthu.edu.tw/~chungchichen/FinNLP2022_IJCAI/7.pdf)

### Few-shot NER
Few-shot NER은 데이터 부족 및 entity가 꾸준하게 변하는 문제를 해결하기 위해 연구되는 분야.<br>
Finance domain은 아니지만, Few-shot NER이 small sample domain에 적용할 수 있고, Finance data가 이에 해당하기 때문에 작성
- (EMNLP 2021) [Few-Shot Named Entity Recognition: A Comprehensive Study](https://aclanthology.org/2021.emnlp-main.813.pdf)
- (ACL 2021) [Few-NERD: A Few-Shot Named Entity Recognition Dataset](https://aclanthology.org/2021.acl-long.248/)
- (RepL4NLP 2022)[A Comparative Study of Pre-trained Encoders for Low-Resource Named Entity Recognition](https://aclanthology.org/2022.repl4nlp-1.6/)

## Finance-NLP dataset & Corpus
1. Semeval 2017 Task 5
    - paper: https://aclanthology.org/S17-2089/
    - dataset: https://alt.qcri.org/semeval2017/task5/index.php?id=data-and-tools
    - Finance Microblog와 News에 대한 Sentiment dataset
2. Will-They-Won't-They(2020)
    - paper: https://arxiv.org/abs/2005.00388
    - dataset: https://github.com/cambridge-wtwt/acl2020-wtwt-tweets
    - rumor와 관련된 Twitter dataset, healthcare에 대한 종목들만 포함되어 있음
3. Daily News Title dataset
    - https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests?resource=download
4. FiQA 2018 dataset
    - https://sites.google.com/view/fiqa/home
    - WWW 2018에서 Financial Opinion Mining and Questing Answering Open Challenge를 개최하며 데이터셋을 공개
    - Aspect-based financial sentiment analysis, Opinion-based QA over financial에 해당하는 데이터셋 있음
5. SEC-BERT 및 Edger corpus
    - https://huggingface.co/nlpaueb/sec-bert-shape
    - https://github.com/nlpaueb/edgar-crawler
        - https://arxiv.org/pdf/2109.14394.pdf
        - SEC 10-K을 기반으로 학습한 Corpus 및 embedding을 제공하고 있음
        - 1993-2020년의 38,009개 company에 대해 6.5B token을 학습
        - 2022.07.15 기준 가장 최신 SEC File 기반 corpus
6. Finance Numeric Entity dataset(Fi-NER 139)
    - https://huggingface.co/datasets/nlpaueb/finer-139

7. ~~한글 wikipedia model 및 dataset(사라짐)~~
    - ~~GPT2용 모델 및 데이터셋~~
    - ~~https://huggingface.co/datasets/eaglewatch/korean_wikipedia_dataset_for_GPT2~~
8. 금융 관련 데이터 API
    - https://site.financialmodelingprep.com/developer/docs
    - 무료버전도 있긴 하지만 제한없이 사용하려면 유료
9. Financial Earning Conference Calls (ECCs)
    - https://github.com/Alaa-Ah/The-FinArg-Dataset-Argument-Mining-in-Financial-Earnings-Calls
10. ESG 관련 데이터
    - https://mx.nthu.edu.tw/~chungchichen/FinNLP2022_IJCAI/9.pdf
        - FinNLP에서 챌린지로 오픈한 데이터셋
        - ESG concept 분류, sustainable/unsustainable을 분류하는 task가 포함되어 있음
        - 데이터의 양은 많지 않음
11. Aspect-Based Sentimetn Analyis
    - https://www.kaggle.com/datasets/ankurzing/aspect-based-sentiment-analysis-for-financial-news
    - 10,700개 정도의 news headline 데이터와 라벨링이 있으며, 그 중 2,800개 정도는 multi-labeling이 되어 있음
    - 라벨링은 (Aspect, sentiment) 로 구성되어 있음
12. 국내 News 데이터
    - https://www.bigkinds.or.kr/v2/news/index.do
    - 위 사이트에서 옵션을 설정하면 뉴스들을 조회할 수 있고, 최대 2만개 데이터에 대해 다운로드 가능
13. 나무 위키 데이터
    - title-sentence 형태로 구성되어 있는 것으로 보임
    - 데이터: https://huggingface.co/datasets/heegyu/namuwiki-extracted
    - 파싱: https://github.com/jonghwanhyeon/namu-wiki-extractor
14. 머니스테이션
    - stocktwiw의 국내 버전으로 보임
    - 국내 유동성의 문제? 혹은 아직 유명하지 않은 문제로 인해 피드가 많지는 않음
    - 종목, sentiment가 각 피드에 태깅되어 있음
    - https://www.moneystation.net/main
15. Financial News Headlines
    - https://www.kaggle.com/datasets/notlucasp/financial-news-headlines
    - 약 2017~2020년까지의 News 헤드라인
16. KorFin-ABSA
    - 한국어 금융 ABSA 데이터셋, 약 15000개 정도의 sample이 있음
    - https://huggingface.co/datasets/amphora/korfin-asc
17. Financial News Topic
     - https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic
18. Twit sentiment
    - https://github.com/jpmcair/tweetfinsent
    - JP Morgan에서 만든 데이터
    - Tweet ID만 있는데, Twitter API를 통해 원문을 수집해야 함
19. FLANG
    - https://github.com/SALT-NLP/FLANG
https://aclanthology.org/2022.emnlp-main.148.pdf
    - JP Morgan에서 만든 finance domain 모델
    - 여러 Dataset도 다 모아뒀음
    - ELECTRA 모델은 뭔가 문제가 있어서 load가 불가능하고, 사용시 약간의 fine-tuning을 진행하면 되는듯
        - 약 2 epoch 정도?

