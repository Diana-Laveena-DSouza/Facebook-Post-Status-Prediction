#1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn import utils
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
import pickle

#2. Load Train datasets
train_1 = pd.read_csv("C:/Users/lloyd/Desktop/iNeuron/Facebook_Train/output-000001.csv")
train_2 = pd.read_csv("C:/Users/lloyd/Desktop/iNeuron/Facebook_Train/output-000002.csv")
train_3 = pd.read_csv("C:/Users/lloyd/Desktop/iNeuron/Facebook_Train/output-000003.csv")
train_4 = pd.read_csv("C:/Users/lloyd/Desktop/iNeuron/Facebook_Train/output-000004.csv")

#Load Validation datasets
val_1 = pd.read_csv("C:/Users/lloyd/Desktop/iNeuron/Facebook_Val/output-000001.csv")
val_2 = pd.read_csv("C:/Users/lloyd/Desktop/iNeuron/Facebook_Val/output-000002.csv")
val_3 = pd.read_csv("C:/Users/lloyd/Desktop/iNeuron/Facebook_Val/output-000003.csv")
val_4 = pd.read_csv("C:/Users/lloyd/Desktop/iNeuron/Facebook_Val/output-000004.csv")

#Concatenate the files
train_data = pd.concat([train_1, train_2, train_3, train_4], axis = 0)
val_data = pd.concat([val_1, val_2, val_3, val_4], axis = 0)
print(train_data.head())

#Size of the data
print("shape of the train data", train_data.shape)
print("shape of the validation data", val_data.shape)

#3. EDA Analysis
#Target Label Counts
print("Target Label Counts")
print(train_data.Emotions.value_counts())


#Checking the Missing Values
print("Number of Missing values in train data:\n", train_data.isna().sum())
print("Number of Missing values in validation data:\n", val_data.isna().sum())

#Text Analytics
posts = " ".join(train_data.Post_Text.values)
wordcloud = WordCloud(max_font_size=50, max_words=30, background_color="white").generate(posts)
#Display the generated image
plt.figure(figsize = (13, 7))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()

#4. Text Preprocessing
#Functions for Text Preprocessing
def text_preprocess(text_data):

    # Lower the text
    text_lower = [text.lower() for text in text_data]

    #Remove hyperlink
    text_hyper_rem = [re.sub(r'http.*.com', ' ', text) for text in text_lower]

    #Remove @name
    text_at_rem = [re.sub(r'@[a-z]+', ' ', text) for text in text_hyper_rem]

    #Remove Special Character
    text_spl_rem = [re.sub(r'[^a-zA-Z]+', ' ', text) for text in text_at_rem]

    #Word Tokenize
    text_token = [word_tokenize(text) for text in text_spl_rem]

    #Remove Stop Words and Stem
    stop_words = list(set(stopwords.words('english'))-{'not', 'no'})
    lemm = WordNetLemmatizer()
    text_lem = []
    for text in text_token:
        text_lem.append([lemm.lemmatize(word) for word in text if word not in stop_words])

    return text_lem

#Call a Function
train_final = text_preprocess(train_data.Post_Text.values)
val_final = text_preprocess(val_data.Post_Text.values)

#Label Encoder
label_trans = LabelEncoder().fit(train_data.Emotions.values)
train_data['Emotions'] = label_trans.transform(train_data.Emotions.values)
val_data['Emotions'] = label_trans.transform(val_data.Emotions.values)

#Create Train Documents, Validation Documents
train_documents = [TaggedDocument(words = x, tags = [y]) for x, y in zip(train_final, train_data.Emotions.values)]
val_documents = [TaggedDocument(words = x, tags = [y]) for x, y in zip(val_final, val_data.Emotions.values)]

#Doc2Vec Model
model = Doc2Vec(vector_size = 3500, min_count=3, epochs=30)
model.build_vocab([x for x in train_documents])
train_documents = utils.shuffle(train_documents)
model.train(train_documents, total_examples = len(train_documents), epochs = model.epochs)

#Function for Vector for Learning
def vector_for_learn(model, data):
    target, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in data])
    return target, feature_vectors

#Call a Function
Y_train, X_train = vector_for_learn(model, train_documents)
Y_val, X_val = vector_for_learn(model, val_documents)

#OverSampling
ros = RandomOverSampler(random_state = 10)
X_train_sam, Y_train_final = ros.fit_resample(X_train, Y_train)

#Feature Scaling
st_sc = StandardScaler().fit(X_train_sam)
X_train_final = st_sc.transform(X_train_sam)
X_val_final = st_sc.transform(X_val)

#Shapes of the training and Validation Data
print("X_train: ", np.array(X_train_final).shape, " X_val: ", np.array(X_val_final).shape, "Y_train: ", np.array(Y_train_final).shape, " Y_val: ", np.array(Y_val).shape)

#5. Model Building and Evaluation
#1. Model Selections
def Classifier(classifier_model, x_train, y_train, x_val, y_val):
    model = classifier_model
    train_model = model.fit(x_train, y_train)
    val_pred = train_model.predict(x_val)
    return (accuracy_score(y_val, val_pred), classification_report(y_val, val_pred))

#Random Forest Classifier
val_acc, class_rep = Classifier(RandomForestClassifier(max_depth = 6), X_train_final, Y_train_final, X_val_final, Y_val)
print("Validation Accuracy using Random Forest Classifier: ", val_acc)
print("Classification Report using Random Forest Classifier: \n", class_rep)


#SVC
val_acc, class_rep = Classifier(SVC(), X_train_final, Y_train_final, X_val_final, Y_val)
print("Validation Accuracy using Support Vector Classifier: ", val_acc)
print("Classification Report using Support Vector Classifier: \n", class_rep)


#Logistic Regression
val_acc, class_rep = Classifier(LogisticRegression(max_iter = 3000, multi_class = 'multinomial'), X_train_final, Y_train_final, X_val_final, Y_val)
print("Validation Accuracy using Logistic Regression: ", val_acc)
print("Classification Report using Logistic Regression: \n", class_rep)


#KNeighborsClassifier
val_acc, class_rep = Classifier(KNeighborsClassifier(n_neighbors = 15), X_train_final, Y_train_final, X_val_final, Y_val)
print("Validation Accuracy using KNeighborsClassifier: ", val_acc)
print("Classification Report using KNeighborsClassifier: \n", class_rep)

#2. Best Model - SVC
svc_model = SVC().fit(X_train_final, Y_train_final)
val_pred = svc_model.predict(X_val_final)
print("Validation Accuracy: ", accuracy_score(Y_val, val_pred))

#Save the Models
pickle.dump(svc_model, open('svc_model.pkl', 'wb'))
pickle.dump(model, open('doc2vec_model.pkl', 'wb'))
pickle.dump(st_sc, open('stsc_model.pkl', 'wb'))