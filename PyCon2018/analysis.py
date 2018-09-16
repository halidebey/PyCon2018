#####ACQUIRE DATA
###Download dataset from https://www.figure-eight.com/data-for-everyone/
# search for: 'Disasters on social media'
# Contributors looked at over 10,000 tweets culled with a variety of searches
# like “ablaze”, “quarantine”, and “pandemonium”,
# then noted whether the tweet referred to a disaster event
# (as opposed to a joke with the word or a movie review or something non-disastrous)


#####EXPLORE DATA
import pandas
input_file = pandas.read_csv("socialmedia-disaster-tweets-DFE.csv",encoding='latin-1')

input_file.shape
input_file.head(50)
input_file.tail(50)

input_file.columns()
#looks like 'tweet' column has the tweet
#and 'choose_one' has the classification

#how many unique values are in choose_one?
input_file.choose_one.unique()
#'Relevant', 'Not Relevant', "Can't Decide"


#####CLEAN UP DATA
#we are only interested in two columns...
input_file=input_file[["text", "choose_one"]]
input_file["choose_one"]=input_file.choose_one.replace({"Relevant": 1, "Not Relevant": 0})
input_file.rename(columns={"choose_one":"label"}, inplace=True)

input_file.label=pandas.to_numeric(input_file.label, errors='coerce')
input_file.dropna(inplace=True)
#Check that data looks as expected
input_file.label.unique()

input_file["text"] = input_file["text"].str.replace(r"http\S+|http|@\S+|at", "")
input_file["text"] = input_file["text"].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
input_file["text"] = input_file["text"].str.lower()


input_file.head(50)
input_file.tail(50)


#####TOKENIZE
#This is the process of turning sentences into a list of words
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
input_file["tokens"] = input_file["text"].apply(tokenizer.tokenize)

#tokens can give us more insight into the data
all_words = [word for tokens in input_file["tokens"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in input_file["tokens"]]
vocabulary = sorted(set(all_words))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(vocabulary)))
print("Max sentence length is %s" % max(sentence_lengths))


#####EMBED
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

text = input_file["text"].tolist()
labels = input_file["label"].tolist()
X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=0.2,random_state=40)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)


#####CHOOSE A CLASSIFIER
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', multi_class='multinomial', n_jobs=-1, random_state=40)
classifier.fit(X_train_tfidf, y_train)

y_predicted_tfidf = classifier.predict(X_test_tfidf)

#####EVALUATE
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_predicted_tfidf, pos_label=None,average='weighted')
print(precision)

