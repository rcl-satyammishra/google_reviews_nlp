import streamlit as st
import nltk, spacy
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
import matplotlib as mpl
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import seaborn as sns
from streamlit import components

st.set_page_config(layout="wide")
service_provider = st.selectbox(
    "Select Diagnostic Service Provider",
    ("Redcliffe Labs", "Healthians", "Lal PathLabs")
)

# nlp = spacy.load("en_core_web_sm")
from st_utils import *


@st.cache(hash_funcs={"MyUnhashableClass": lambda _: None})
def read_data():
    if service_provider == "Redcliffe Labs":
        redcliffe_labs = pd.read_excel('redcliffelabs15k.xlsx', parse_dates=['review_datetime_utc'])

    elif service_provider == 'Lal PathLabs':
        redcliffe_labs = pd.read_excel('lalpathlabs.xlsx', parse_dates=['review_datetime_utc'])
    else:
        redcliffe_labs = pd.read_csv('healthians_1k_recent_new.csv', parse_dates=['review_datetime_utc'])
    redcliffe_labs = redcliffe_labs.dropna()
    redcliffe_labs = redcliffe_labs[['review_text', 'review_rating', 'review_datetime_utc']]
    redcliffe_labs['review_text'] = pd.DataFrame(redcliffe_labs.review_text.apply(lambda x: clean_text(x)))
    # redcliffe_labs["review_text"] = redcliffe_labs.apply(lambda x: lemmatizer(x['review_text']), axis=1)
    redcliffe_labs['review_lemmatize_clean'] = redcliffe_labs['review_text'].str.replace('-PRON-', '')
    redcliffe_labs['polarity'] = redcliffe_labs.review_lemmatize_clean.apply(detect_polarity)
    redcliffe_labs = redcliffe_labs.sort_values(by='review_datetime_utc')
    redcliffe_labs['keywords'] = redcliffe_labs.review_text.apply(search_service)
    return redcliffe_labs


redcliffe_labs = read_data()

st.title(service_provider + ' Google Reviews')

with st.expander("See Data"):
    st.write(redcliffe_labs.head())

my_slider = st.checkbox('Select Reviews with Polarity Values', True)
if my_slider:
    st.subheader('Select Reviews with Polarity Values')
    with st.form(key='my_form'):
        values = st.slider(
            'Select a range of Review Polarity values',
            -1.0, 1.0, (-1.0, -0.3))
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write('Selected Polarity Values:', values)
            st.write(redcliffe_labs[['polarity', 'review_text', 'review_rating']][
                         redcliffe_labs['polarity'].between(values[0], values[1])])
        else:
            st.write(redcliffe_labs[['polarity', 'review_text', 'review_rating']][
                         redcliffe_labs['polarity'].between(-1.0, 0.51)])

c1, c2 = st.columns(2)
rating_polarity = redcliffe_labs.groupby([redcliffe_labs['review_datetime_utc'].dt.month_name()],
                                         sort=False).mean().reset_index()
fig = px.line(rating_polarity, x="review_datetime_utc", y="review_rating")
with c1:
    st.plotly_chart(fig, use_container_width=True)
    fig_2 = px.line(rating_polarity, x="review_datetime_utc", y="polarity")
with c2:
    st.plotly_chart(fig_2, use_container_width=True)

month_trend = st.checkbox('Month wise Trend', True)
if month_trend:
    sdf_ = redcliffe_labs.groupby([redcliffe_labs['review_datetime_utc'].dt.month_name(), redcliffe_labs['keywords']],
                                  sort=False).agg(['count', 'mean'])[
        ['polarity']].reset_index()
    sdf_.columns = ['month', 'keyword', 'polarity_count', 'polarity_mean']
    if service_provider == "Redcliffe Labs":
        sdf_ = sdf_[sdf_['polarity_count'] > 20]
    fig = px.line(sdf_, x="month", y="polarity_mean", color='keyword', markers=True)
    fig.update_layout(
        autosize=False,
        width=1200,
        height=600, )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(sdf_, x='month', y='polarity_count', color='keyword')
    fig.update_layout(barmode='group')
    st.plotly_chart(fig, use_container_width=True)

v_polarity = st.checkbox('Show Review Polarity/Rating Plots ', True)
st.subheader('Polarity Values Plots ')

col1, col2 = st.columns(2)
with col1:
    if v_polarity:
        # A histogram of the polarity scores.
        num_bins = 50
        fig_pol = plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(redcliffe_labs.polarity, num_bins, facecolor='blue', alpha=0.5)
        plt.xlabel('Polarity')
        plt.ylabel('Count')
        # plt.title('Histogram of polarity')
        st.pyplot(fig_pol)
with col2:
    fig_rat = plt.figure(figsize=(10, 6))
    sns.boxenplot(x='review_rating', y='polarity', data=redcliffe_labs)
    st.pyplot(fig_rat)

st.info(
    'The more a specific word appears in a source of reviews, the bigger and bolder it appears in the word cloud.')
col_1, col_2 = st.columns(2)
with col_1:
    positive_wordcloud = st.checkbox('Visualize Positive WordCloud', True)
    if positive_wordcloud:
        st.subheader('Positive WordCloud')
        mpl.rcParams['figure.figsize'] = (12.0, 12.0)
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['savefig.dpi'] = 100
        mpl.rcParams['figure.subplot.bottom'] = .1
        stopwords = set(STOPWORDS)

        wordcloud = WordCloud(
            background_color='white',
            stopwords=stopwords,
            max_words=500,
            max_font_size=40,
            random_state=42
        ).generate(str(redcliffe_labs[redcliffe_labs['review_rating'].isin([4, 5])]['review_text']))

        print(wordcloud)
        fig_21 = plt.figure(1)
        plt.imshow(wordcloud)
        plt.axis('off')
        st.pyplot(fig_21)

with col_2:
    negative_wordcloud = st.checkbox('Visualize Negative WordCloud', True)
    if negative_wordcloud:
        st.subheader('Negative WordCloud')
        mpl.rcParams['figure.figsize'] = (12.0, 12.0)
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['savefig.dpi'] = 100
        mpl.rcParams['figure.subplot.bottom'] = .1
        stopwords = set(STOPWORDS)

        wordcloud = WordCloud(
            background_color='white',
            stopwords=stopwords,
            max_words=500,
            max_font_size=40,
            random_state=42
        ).generate(str(redcliffe_labs[redcliffe_labs['review_rating'].isin([1, 2])]['review_text']))

        print(wordcloud)
        fig_21 = plt.figure(1)
        plt.imshow(wordcloud)
        plt.axis('off')
        st.pyplot(fig_21)
# wordcloud = WordCloud(
#     background_color='white',
#     stopwords=stopwords,
#     max_words=500,
#     max_font_size=40,
#     random_state=42
# ).generate(str(redcliffe_labs['review_text'][]))
#
# print(wordcloud)
# fig = plt.figure(1)
# plt.imshow(wordcloud)
# plt.axis('off')
# st.pyplot(fig)

positive_grams = st.checkbox('Visualize Unigrams, Bigrams and Trigrams in Positive Reviews', True)
if positive_grams:
    st.subheader('Positive Reviews: Most Frequently occurring sequence of N words')
    common_words = get_top_n_words(
        redcliffe_labs[redcliffe_labs['review_rating'].isin([4, 5])]['review_lemmatize_clean'], 30)
    uni_gram = pd.DataFrame(common_words, columns=['unigram', 'count'])

    fig = go.Figure([go.Bar(x=uni_gram['unigram'], y=uni_gram['count'])])
    fig.update_layout(
        title=go.layout.Title(text="Top 30 unigrams in the positive reviews."))
    st.plotly_chart(fig, use_container_width=True)

    common_words = get_top_n_bigram(
        redcliffe_labs[redcliffe_labs['review_rating'].isin([4, 5])]['review_lemmatize_clean'], 20)
    bi_gram = pd.DataFrame(common_words, columns=['bigram', 'count'])
    fig = go.Figure([go.Bar(x=bi_gram['bigram'], y=bi_gram['count'])])
    fig.update_layout(
        title=go.layout.Title(text="Top 20 bigrams in the positive reviews."))
    st.plotly_chart(fig, use_container_width=True)

    common_words = get_top_n_trigram(
        redcliffe_labs[redcliffe_labs['review_rating'].isin([4, 5])]['review_lemmatize_clean'], 20)
    tri_gram = pd.DataFrame(common_words, columns=['trigram', 'count'])
    fig = go.Figure([go.Bar(x=tri_gram['trigram'], y=tri_gram['count'])])
    fig.update_layout(title=go.layout.Title(text="Top 20 trigrams in the positive reviews"))
    st.plotly_chart(fig, use_container_width=True)

negative_grams = st.checkbox('Visualize Unigrams, Bigrams and Trigrams in Negative Reviews', True)
if negative_grams:
    st.subheader('Negative Reviews: Most Frequently occurring sequence of N words')
    common_words = get_top_n_words(
        redcliffe_labs[redcliffe_labs['review_rating'].isin([1, 2])]['review_lemmatize_clean'], 30)
    uni_gram = pd.DataFrame(common_words, columns=['unigram', 'count'])

    fig = go.Figure([go.Bar(x=uni_gram['unigram'], y=uni_gram['count'])])
    fig.update_layout(
        title=go.layout.Title(text="Top 30 unigrams in the negative reviews."))
    st.plotly_chart(fig, use_container_width=True)

    common_words = get_top_n_bigram(
        redcliffe_labs[redcliffe_labs['review_rating'].isin([1, 2])]['review_lemmatize_clean'], 20)
    bi_gram = pd.DataFrame(common_words, columns=['bigram', 'count'])
    fig = go.Figure([go.Bar(x=bi_gram['bigram'], y=bi_gram['count'])])
    fig.update_layout(
        title=go.layout.Title(text="Top 20 bigrams in the negative reviews."))
    st.plotly_chart(fig, use_container_width=True)

    common_words = get_top_n_trigram(
        redcliffe_labs[redcliffe_labs['review_rating'].isin([1, 2])]['review_lemmatize_clean'], 20)
    tri_gram = pd.DataFrame(common_words, columns=['trigram', 'count'])
    fig = go.Figure([go.Bar(x=tri_gram['trigram'], y=tri_gram['count'])])
    fig.update_layout(title=go.layout.Title(text="Top 20 trigrams in the negative reviews"))
    st.plotly_chart(fig, use_container_width=True)
# st.write(uni_gram)
# st.write(bi_gram)
# st.write(tri_gram)


with st.expander("Reviews that have the Highest polarity or Lowest Polarity:"):
    st.write("Reviews that have the Highest polarity:")
    st.write(redcliffe_labs[redcliffe_labs.polarity == 1].review_text.head(25))
    st.write("Reviews that have the lowest polarity:")
    st.write(redcliffe_labs[redcliffe_labs.polarity == -1].review_text.head(25))
    st.write("Reviews that have the lowest ratings:")
    st.write(redcliffe_labs[redcliffe_labs.review_rating == 1].review_text.head(25))
    # st.write("Reviews that have the Highest ratings:")
    # st.write(redcliffe_labs[redcliffe_labs.review_rating == 5].review_text.head())
    # st.write("Reviews that have lowest polarity (most negative sentiment) but with a 5-star:")
    # st.write(redcliffe_labs[(redcliffe_labs.review_rating == 5) & (redcliffe_labs.polarity == -1)].head(10))
    # st.write("Reviews that have the highest polarity (most positive sentiment) but with a 1-star:")
    # st.write(redcliffe_labs[(redcliffe_labs.review_rating == 1) & (redcliffe_labs.polarity == 1)].head(10))

    # st.write(redcliffe_labs.head())

with st.expander("See Distribution of review character length"):
    plt.figure(figsize=(10, 6))
    doc_lens = [len(d) for d in redcliffe_labs.review_text]
    fig, ax = plt.subplots()
    plt.hist(doc_lens, bins=100)
    plt.title('Distribution of review character length')
    plt.ylabel('Number of reviews')
    plt.xlabel('Review character length')
    sns.despine()
    st.pyplot(fig)
#
# if topic_distribution:
#     st.title(service_provider + ' Topic Distribution')
#     # with st.form(key='Words_form'):
#     #     words = int(st.text_input(
#     #         "Number of Words",
#     #         10
#     #     ))
#     #     submitted = st.form_submit_button("Submit")
#     #     if submitted:
#     #         words = words
#     #     else:
#     #         words = 10
#
#     # Topic - Keywords Dataframe
#     df_topic_keywords = pd.DataFrame(topic_keywords)
#     df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
#     topics = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
#     df_topic_keywords.index = topics
#     st.subheader('Topic Keywords Table')
#     st.write(df_topic_keywords.T)
#     lda_output = lda_model.transform(data_vectorized)
#     topicnames = df_topic_keywords.T.columns
#     docnames = ["Doc" + str(i) for i in range(len(redcliffe_labs))]
#     df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
#     dominant_topic = np.argmax(df_document_topic.values, axis=1)
#     df_document_topic['dominant_topic'] = dominant_topic
#     # st.write(df_document_topic)
#     df_document_topic.reset_index(inplace=True)
#     df_sent_topic = pd.merge(redcliffe_labs, df_document_topic, left_index=True, right_index=True)
#     df_sent_topic.drop('index', axis=1, inplace=True)
#     with st.expander("See Topic wise score of each review."):
#         st.write(df_sent_topic)
#     df_topic_theme = df_sent_topic[['review_text', 'dominant_topic']]
#
#     df_topic_distribution = df_topic_theme.groupby(['dominant_topic']).size().sort_values(ascending=False).reset_index(
#         name='count').drop_duplicates()
#     df_topic_distribution['dominant_topic'] = 'Topic ' + df_topic_distribution['dominant_topic'].astype('str')
#     # st.write(df_topic_distribution)
#     fig = px.bar(df_topic_distribution, x='dominant_topic', y='count')
#     st.plotly_chart(fig, use_container_width=True)
#
#     st.subheader('Dominant Topic: Filter')
#     with st.form(key='Filter_form'):
#         topic_ = st.selectbox(
#             "Select Topic",
#             topics
#         )
#         no = int(''.join(filter(str.isdigit, topic_)))
#         submitted = st.form_submit_button("Submit")
#         if submitted:
#             st.write(df_topic_theme[df_document_topic.dominant_topic == no])
#         else:
#             st.write(df_topic_theme)
#
#     # df_topic_theme['dominant_topic_theme'] = df_topic_theme.apply(lambda row: label_theme(row), axis=1)
#
#     # df_topic_distribution = df_topic_theme.groupby(['dominant_topic', 'dominant_topic_theme']).size().sort_values(ascending=False).reset_index(name='count').drop_duplicates(subset='dominant_topic_theme')
st.info('Please reload/reboot if problem occurs due to loading!')
