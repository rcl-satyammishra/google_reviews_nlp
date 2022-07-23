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

# nlp = spacy.load("en_core_web_sm")
from st_utils import *

redcliffe_labs = pd.read_excel('RedcliffeLabs2K.xlsx')
redcliffe_labs = redcliffe_labs.dropna()

redcliffe_labs = pd.DataFrame(redcliffe_labs.review_text.apply(lambda x: clean_text(x)))
# redcliffe_labs["review_text"] = redcliffe_labs.apply(lambda x: lemmatizer(x['review_text']), axis=1)
st.write(redcliffe_labs.head())

plt.figure(figsize=(10, 6))
doc_lens = [len(d) for d in redcliffe_labs.review_text]
fig, ax = plt.subplots()
plt.hist(doc_lens, bins=100)
plt.title('Distribution of review character length')
plt.ylabel('Number of reviews')
plt.xlabel('Review character length')
st.pyplot(fig)

import seaborn as sns

sns.despine()

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
).generate(str(redcliffe_labs['review_text']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
st.pyplot(fig)

redcliffe_labs['review_lemmatize_clean'] = redcliffe_labs['review_text'].str.replace('-PRON-', '')

common_words = get_top_n_words(redcliffe_labs['review_lemmatize_clean'], 30)
uni_gram = pd.DataFrame(common_words, columns=['unigram', 'count'])

fig = go.Figure([go.Bar(x=uni_gram['unigram'], y=uni_gram['count'])])
fig.update_layout(
    title=go.layout.Title(text="Top 30 unigrams in the question text after removing stop words and lemmatization"))
st.plotly_chart(fig, use_container_width=True)

common_words = get_top_n_bigram(redcliffe_labs['review_lemmatize_clean'], 20)
bi_gram = pd.DataFrame(common_words, columns=['bigram', 'count'])
fig_2 = go.Figure([go.Bar(x=bi_gram['bigram'], y=bi_gram['count'])])
fig_2.update_layout(
    title=go.layout.Title(text="Top 20 bigrams in the question text after removing stop words and lemmatization"))
st.plotly_chart(fig_2, use_container_width=True)

common_words = get_top_n_trigram(redcliffe_labs['review_lemmatize_clean'], 20)
tri_gram = pd.DataFrame(common_words, columns=['trigram', 'count'])
fig = go.Figure([go.Bar(x=tri_gram['trigram'], y=tri_gram['count'])])
fig.update_layout(title=go.layout.Title(text="Top 20 trigrams in the question text"))
st.plotly_chart(fig, use_container_width=True)
st.write(uni_gram)
st.write(bi_gram)
st.write(tri_gram)

data_vectorized = vectorizer.fit_transform(redcliffe_labs['review_lemmatize_clean'])
lda_output = lda_model.fit_transform(data_vectorized)

# pyLDAvis.enable_notebook()
prepared_model_data = pyLDAvis.sklearn.prepare(lda_model, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.save_html(prepared_model_data, 'lda.html')

html_string = pyLDAvis.prepared_data_to_html(prepared_model_data)
from streamlit import components

components.v1.html(html_string, width=1300, height=800, scrolling=True)

topic_keywords = show_topics(vectorizer, lda_model, n_words=10)

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
st.write(df_topic_keywords.T)

# Create Document - Topic Matrix
lda_output = lda_model.transform(data_vectorized)

# column names
topicnames = df_topic_keywords.T.columns
# topicnames = ["Topic" + str(i) for i in range(20)]

# index names
docnames = ["Doc" + str(i) for i in range(len(redcliffe_labs))]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic
st.write(df_document_topic)

df_document_topic.reset_index(inplace=True)
df_sent_topic = pd.merge(redcliffe_labs, df_document_topic, left_index=True, right_index=True)
df_sent_topic.drop('index', axis=1, inplace=True)
st.write(df_sent_topic)

df_topic_theme = df_sent_topic[['review_text', 'dominant_topic']]
st.write(df_topic_theme.head(10))

# df_topic_theme['dominant_topic_theme'] = df_topic_theme.apply(lambda row: label_theme(row), axis=1)

# df_topic_distribution = df_topic_theme.groupby(['dominant_topic', 'dominant_topic_theme']).size().sort_values(ascending=False).reset_index(name='count').drop_duplicates(subset='dominant_topic_theme')

df_topic_distribution = df_topic_theme.groupby(['dominant_topic']).size().sort_values(ascending=False).reset_index(name='count').drop_duplicates()
st.write(df_topic_distribution)
