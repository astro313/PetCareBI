# PetCareBI
A machine learning platform for extracting insights from customer reviews in the pet service industry. Try it: [Front-end deployed on AWS](http://www.future-analytics.me:8501)

## Motivation
68% of US household owns a pet and the annual cash flow of the pet service industry is >$10B, which is projected to continue its growth in 2020 despite COVID-19. To help business owners to stand out from their competitors and promote customer retention in this lucrative industry, PetCare BI is business intelligence tool built for increasing customer satisfaction score and reducing the number of poor reviews. 

Reviews are often overwhelming in length, with information buried in unstructured text, written in different styles, and cover different aspects in each. PetCare BI provides a dashboard showing the key topics discussed in Yelp reviews, topic trends over the years, and over each review rating group. It  uses AI to provide executive summary to help business owners quickly gauge consumers’ feedback on services. With PetCare BI, time spent on reading and understanding reviews can be reduced by >50%! 

### Based on these insights, business owners can:
    1. determine which aspects of the business is having bigger issues with customers. 
    2. identify your USP (Unique selling proposition) to promo a brand that stands out from your competitors and enhance customer retention.
    3. improve service based on customer reviews and incentivize "turned away" customers --> updated review to attract more customers.
    4. strategize their marketing effort or business model based on latest trends (e.g., providing updates on pets in day care/boarding service).


### Example 
<img src="demo/rating.png" width="50%">
<img src="demo/pie.png" width="50%">
<img src="demo/trend.png" width="50%">

#### Project Aim:
This project is divided into 6 modules.
* **Part I: Web_scraping**
    - Obtaining Yelp review for businesses in pet service industry. Specifically in pet kennels, hotels, grooming, and day care. Scraped using selenium and beautifulsoup.
* **Part II: synthesize data, numerical and visual EDA**
    - Basic data management, combining files, removing duplicates, NA rows, etc
* **Part III: NLP**
    - URL, numbers, stopwords, keep nouns, lemmatization, POS identification, etc
* **Part IV: LDA modeling**
    - hyperparmeter turning
    - validation
    - visualization
* **Part V: Text summarization** 
    - Abstractive summary: AI-based (transformer model)
    - Extractive summary: frequency based (NLTK)
* **Part VI: Front-end/Results** 
    - Docker + streamlit + AWS 


### Technical aspects
PetCare BI is based on Latent Dirichlet Allocation (LDA) model and SOTA T5 transformer model. 


#### Note/challenges/edge cases:
- Choosing right algorithm for the right job: LDA vs. embedding + clustering, etc.  
    1. chose LDA because interested in thematic topics/clusters, not semantic & syntactic groupings (e.g., good reviews and great reviews). Clustering embedding yields the latter. 
    2. Not use NMF because large DS + topic probabilities unlikely fixed per review
- Low coherence: many redundant words (useless feature vectors) → Went back to EDA, understand what does “topic” mean → Keeping only nouns + remove  customized words + understand & tuned hyperparameters (# of topics, alpha, beta)
- Interesting insight from data : trend of day care/sitter w/ updates and photos
- LDA doesn’t scale very well (training time for one epoch is proportional to scale of dataset). Luckily, number of topics doesn’t change that often -- don’t need to retrain model too frequently. 
- Edge cases: in topic modeling, sometimes not very good at determining which is dominant topic in the review. → so we also provide text summary and aspect-based sentiment (todo).
- ways to validate result.
- Overall, understanding the algorithm and developing a heuristic approach to clean the data and perform NLP was key to getting good results in topic modeling. 


#### To Launch webapp on EC2 using saved LDAmodel and dataset
1. `git clone git@github.com:astro313/MyPetCare.git`
2. `cd MyPetCare`
3. unzip LDAmodel.tgz and data.tgz
4. `docker image build -t streamlit:app .`
5. `docker container run -p 8501:8501 --name petcareBI -it -d streamlit:app bash`
6. `tmux new -s StreamlitWebApp`
7. `docker exec -it petcareBI bash`
8. `streamlit run webapp/app.py`


#### To run from scratch to LDA_momdel part:
if scrape data:

    1. save zipcodes into .txt files in data/
    2. scrape_biz_link_multipage.py
    3. scrape_reviews.py
    4. zipcode_based_drop_dupl_biz.py
if create single pd dataframe w/ all reviews for creating new model:

    1. df = load_review_helper.combine_all_reviews_for_model()
    2. df_new = NLP_cleaning.apply_NLP_cleaning(df_new)
    3. df_new.to_csv('data_model/cleaned_tokenized_df-2020-....csv', index=False)
if load cleaned dataframe only and use existing model:

    1. df_new = pd.read_csv('data_model/cleaned_tokenized_df-2020-....csv')
    2. lda_model = streamlitapp.load_pretrain_ldamodel(fname=r'data_model/.....model')
if train new model:

    if load review from saved:
        1. tmp = df_new['review_text_lem_cleaned_tokenized_nostop'].apply(lambda x: NLP_cleaning.prep_lda_input(x)).tolist()
    else if create dataframe with all reviews on-the-fly
        1. tmp = df_new['review_text_lem_cleaned_tokenized_nostop']
    2. dictionary, bow_corpus =  ldacomplaints.get_dict_and_corpus(tmp)
    3. lda_model, df_dominant_topic, topic_contribution = ldacomplaints.lda_analysis(tmp, num_topics=10)
