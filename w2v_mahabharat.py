#########################################Word2Vec on Mahabharat#################################################3



#importing libraries
from zipfile import ZipFile
import codecs
import glob
import nltk
nltk.download('punkt')
import io
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import stopwords
import gensim.models.word2vec as w2v
nltk.download('stopwords')


#Reading text files
data_path = 'mb.txt'
with open(data_path, 'r') as f:
    lines = f.read().split('\n')

# print(lines[:10])

#Converting text corpus into string
corpus=''
for i in lines:
    corpus+=i
# print(corpus[:50])

#Performing sentence tokenization
sentences=sent_tokenize(corpus)

#Sentence cleaning function
def clean_sent(raw):
    clean=re.sub('[^a-zA-Z]',' ',raw)
    words=clean.split()
    return words

raw_sent=[clean_sent(i) for i in sentences]

#Defining set of stopwords
stop_words = set(stopwords.words('english'))


sent=[]
stop=list(stopwords.words('english'))
# print(stop)

#Performing preprocessing of text
for i in raw_sent:
    blank=[]
    
    for j in i:
        if j.lower() not in stop:
            blank.append(j)
    sent.append(blank)

#Defining structure of model
model=w2v.Word2Vec(sg=1,
    seed=0,
    workers=8,
    size=100,
    min_count=5,
    window=12,
    iter=5
    )

#Creating vocabulary 
model.build_vocab(sent)
token_count=len(model.wv.vocab)

#Training model on corpus
model.train(sent, total_words=token_count, epochs=5)

#Saving vocabulary ,you can use words of vocabulary to play with word2vec
#Word2Vec won't work on words which are not in the vocabulary
with open('mb_vocab.txt','w') as f:
    f.write(str(model.wv.vocab))



#Checking similarity between two words
model.wv.similarity('Arjuna','arrows')

#Finding top 10 most similar words to Arjuna
model.most_similar('Arjuna')



##################################Outputing vocabulary in form of graph#########################################333
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet
output_notebook()
 
X = []
for word in model.wv.vocab:
    X.append(model.wv[word])
 
X = np.array(X)
print("Computed X: ", X.shape)
X_embedded = TSNE(n_components=2, n_iter=250, verbose=2).fit_transform(X)
print("Computed t-SNE", X_embedded.shape)
 
df = pd.DataFrame(columns=['x', 'y', 'word'])
df['x'], df['y'], df['word'] = X_embedded[:5000,0], X_embedded[:5000,1], list(model.wv.vocab.keys())[:5000]
 
source = ColumnDataSource(ColumnDataSource.from_df(df))
labels = LabelSet(x="x", y="y", text="word", y_offset=8,
                  text_font_size="8pt", text_color="#555555",
                  source=source, text_align='center')
 
plot = figure(plot_width=800, plot_height=800)
plot.circle("x", "y", size=12, source=source, line_color="black", fill_alpha=0.8)
plot.add_layout(labels)
show(plot)




