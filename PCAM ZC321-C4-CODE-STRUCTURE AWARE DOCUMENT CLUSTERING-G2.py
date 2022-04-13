#!/usr/bin/env python
# coding: utf-8

# #  PCAM ZC321-C4-CAPSTONE PROJECT 
# # STRUCTURE AWARE DOCUMENT CLUSTERING
# ##  (Group 2 - 2020AIML544, 2020AIML527, 2020AIML567, 2020AIML596)

# ## Overall Approach
# 
#     - Convert all PDFs, DOCXs => JPGs
#     - Draw Bounding boxes and extract text within the bounding boxes using easyOCR
#     - Extract the dimensions of the bounding boxes (x,y,w,h coordinates in form of vectors)
#     - Text pre-processing -> removal of stop words, conversion to lowercase, tokenization, lemmatization
#     - Appended the content to the bounding box vectors
#       (constructed custom dictionary for extracting labels and appended the labels to vectors)
#     - Rearrangement of vectors based on content similarity (used TFidf vectorizer & cosine similarity)
#     - Padding technique to bring similar content at one vector position and post padding 
#     - Normalization of bounding box dimensions to bring all bounding boxes to similar structure
#     - Measure section wise content similarity using TF-IDF vectorizer & cosine similarity for the sake of clustering
#   
#     Clustering:
#       a) Cluster resumes based on structural similarity after normalization 
#           (used K Means clustering - Euclidean distance L2 norm)
#       b) Cluster based on section wise content similarity 
#           (used K Means clustering - TFIDF vectorizer, cosine similarity)
#       c) Cluster using only 1 vector (professional experience) 
#           (used K Means/K Means++ scikit library to cluster)

# #### Install & Import Required Libraries

# In[1]:


# !pip install docx2pdf
# !pip install tesseract
# !pip install easyocr 
# !pip install opencv-python-headless
# !pip install Pillow
# !pip install spacy


# In[2]:


import pandas as pd
import numpy as np
import glob, os, cv2
import matplotlib.pyplot as plt
import spacy
import nltk
import texthero as hero

import re
import easyocr
import PIL
import copy

from pdf2jpg import pdf2jpg
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import silhouette_score
from texthero import preprocessing
from spacy import displacy
from PIL import Image, ImageDraw

spacy.load('en_core_web_sm')
nltk.download('stopwords')
nltk.download('wordnet')

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# #### Read pdf & docx files and convert them to jpg type 

# #### Convert docx files to pdf format

# In[3]:


my_path = "Testresumes/"
for file in os.listdir(my_path):
   if file.endswith(".docx"):
        print(file)


# In[4]:


from docx2pdf import convert
convert(my_path)


# #### Now, convert all pdfs to jpg type

# In[5]:


for file in os.listdir(my_path):
    if file.endswith(".pdf"):
        print(my_path+file)
        result = pdf2jpg.convert_pdf2jpg(my_path+file, "Testresumes/", pages="ALL")


# #### Creation of custom dictionary for extracting labels from the bounding boxes

# In[6]:


test_dict = {'Personal'    : {'name'        : ['name','last name','first name'], 
                              'contact'     : ['phone','mobile','email','residence','location'],
                              'age'         : ['age','dob'],
                              'status'      : ['marital'],
                              'gender'      : ['gender'],
                              'national'    : ['nationalilty','nation','residency','race','citizenship'],
                              'birth'       : ['birth','dob'],
                              'salary'      : ['salary','pay'],
                              'notice'      : ['notice','period','time']
                             },
             'Professional': {'experience'  : ['professional','experience', 'work', 'responsibility','responsible','corporate','career'], 
                              'job'         : ['job','role','position','company','employment','duty','present','past','duration','description'],
                              'profile'     : ['profile','background']
                             },
             'Achievements': {'achievements': ['achievement','achievements','key','award']                 
                             },
             'Education'   : {'education'   : ['education','university','college','institute','institution','academic','qualification','curricular'],
                              'skills'      : ['skills','it','language','fluent','proficient','proficiency','interest','communication','analytical','competency']
                             }
            }


# #### Function for text pre-processing  

# In[7]:


stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text1 = re.sub('[^a-zA-Z \n]', '', text)
    text_lower = text1.lower()
    text_tokens = word_tokenize(text_lower)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    
    wordlist=[]
    for word in tokens_without_sw:
        word = lemmatizer.lemmatize(word)
        wordlist.append(word)
    tokens_without_sw = wordlist

    return tokens_without_sw   


# #### Function to Extract labels from bounding boxes using custom dictionary defined above

# In[8]:


def extractlabels(bow):
    bowlist = []
    key_list = list(test_dict.keys())
    val_list = list(test_dict.values())
    print('key_list',key_list)
    
    for i in range(0,len(bow)):
        for sub1 in test_dict.keys():
            for k in range(0,len(test_dict[sub1].values())):
                len1 = len(test_dict[sub1].values())
                if bow[i] in list(test_dict[sub1].values())[k]:
                    bowlist.append(sub1)
    return bowlist


# #### Read all the images from the path

# In[9]:


ims = []
im  = []
img = 0
for filename in os.listdir(my_path):
    files = glob.glob(my_path + filename + '/*.jpg', recursive=True)
    for file in files:
        if file.endswith(".jpg"):
            img = PIL.Image.open(file)
            im.append(img)
            img = 0
    if len(im) > 0:
        ims.append(im)
    im = []


# #### Functions to draw bounding boxes using easyOCR, extract box dimensions & prepare vector

# In[10]:


reader = easyocr.Reader(['en'])
def draw_boxes(image, bounds, color='red',width=3):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0],fill=color,width=width)
    return image


# In[11]:


def get_vector(bound):
    vector = []
    bowlist = []
    uniquebow = []
    x = bound[0][0][0]
    y = bound[0][0][1]
    w = bound[0][2][0] - x
    h = bound[0][2][1] - y
    text = bound[1]

    if (re.search('[a-zA-Z]', text)) == None:
        print('text is null',text)

    processedtext = preprocess_text(text)
    bowlist       = extractlabels(processedtext)
    indexes = np.unique(bowlist,return_index=True)[1]
    uniquebow = [bowlist[index] for index in sorted(indexes)]
    print('uniquebow',uniquebow)
    vectoro       = [x,y,w,h,text]
    vector        = [x,y,w,h,uniquebow]

    return vector, vectoro


# #### Consider a sample from the image list for training purpose

# In[12]:


imgs = ims
len(imgs)


# #### Loop through the images, draw bouding boxes & get vectors with preprocessed text

# In[13]:


bboxlist, vectorlist  = [], []
bboxlisto, vectorlisto= [], []

for i in range(0,len(imgs)):
    for img in imgs[i]:
        bounds = reader.readtext(img,paragraph="True",contrast_ths=0.05,adjust_contrast=0.7,add_margin=0.35,width_ths=0.5,decoder='wordbeamsearch')
        img = draw_boxes(img,bounds)
    
        for bound in bounds:
            vector, vectoro = get_vector(bound)
            if vector[4] != []:
                vector[4] = vector[4][0]
                vectorlist.append(vector)
                vectorlisto.append(vectoro)      
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        plt.show()
    bboxlist.append(vectorlist)
    print('bboxlist',bboxlist)
    bboxlisto.append(vectorlisto)  
    vectorlist =[]
    vectorlisto=[]


# #### Convert vectors list into df for understanding the layout

# In[14]:


df = pd.DataFrame(bboxlist)
df.columns= ["V"+str(i) for i in range(df.shape[1])]
df.head()


# #### Append the original content to the vector after label for processing

# In[15]:


for i in range(0,len(bboxlist)):
    for j in range(0,len(bboxlist[i])):
        bboxlist[i][j].append(bboxlisto[i][j][4])


# In[16]:


df = pd.DataFrame(bboxlist)
df.columns= ["V"+str(i) for i in range(df.shape[1])]
df.head()


# #### Sort the list on labels 

# In[17]:


for i in range(0,len(bboxlist)):
    bboxlist[i].sort(key=lambda x: (x[4]))


# In[18]:


df = pd.DataFrame(bboxlist)
df.columns= ["V"+str(i) for i in range(df.shape[1])]
df


# #### Merge vectors of same label

# In[19]:


for i in range(0,len(bboxlist)):
    k =0
    j =1
    while k < len(bboxlist[i]):
        if j < len(bboxlist[i]):
            if bboxlist[i][k][4] == bboxlist[i][j][4]:
                bboxlist[i][k][2] = max(bboxlist[i][k][2],bboxlist[i][j][2])
                bboxlist[i][k][3] = bboxlist[i][k][3]+bboxlist[i][j][3]
                bboxlist[i][k][5] = bboxlist[i][k][5]+bboxlist[i][j][5]
                bboxlist[i][j]    = [0,0,0,0,['null']]               
                j = j + 1        
            else:
                k = k+1

                if bboxlist[i][k] == [0,0,0,0,['null']]:
                    k = k + 1       
                    j = k + 1
                    continue
                j = j + 1
        else:
            k = j
            


# In[20]:


df = pd.DataFrame(bboxlist)
df.columns= ["V"+str(i) for i in range(df.shape[1])]
df.head()


# #### Now, split the list into 2 lists with label & original content

# In[21]:


bboxlist1 = copy.deepcopy(bboxlist)
bboxlisto = copy.deepcopy(bboxlist)

for i in range(0,len(bboxlist)):
    for j in range(0,len(bboxlist[i])):
        if bboxlist1[i][j][4] != ['null']:
            c1 = bboxlist1[i][j].pop(5)
            c2 = bboxlisto[i][j].pop(4)


# In[22]:


df = pd.DataFrame(bboxlist)
df.columns= ["V"+str(i) for i in range(df.shape[1])]
df


# In[23]:


bb  = copy.deepcopy(bboxlist1)
bbo = copy.deepcopy(bboxlisto)


# #### Remove null values from the lists

# In[24]:


def remove_values_from_list(the_list, val):
       return [value for value in the_list if value != val]

for i in range(0,len(bb)):
    bb[i] = remove_values_from_list(bb[i], [0, 0, 0, 0, ['null']])
    bbo[i] = remove_values_from_list(bbo[i], [0, 0, 0, 0, ['null']])


# In[25]:


df = pd.DataFrame(bb)
df.columns= ["V"+str(i) for i in range(df.shape[1])]
df


# In[26]:


dfo = pd.DataFrame(bbo)
dfo.columns= ["V"+str(i) for i in range(dfo.shape[1])]
dfo


# #### Measure content similarity & rearrange vectors  

# In[27]:


# Create functions for tfidf similarity 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents1    = []

def listToStr(list):
    listToStr = ' '.join(map(str, list))
    return listToStr
    
def process_tfidf_similarity(base_document,documents):
    documents1 = []
    vectorizer = TfidfVectorizer()
    
    base_str      = f'"{listToStr(base_document)}"'
    documents_str = listToStr(documents)
    documents1.insert(0,f'"{documents_str}"')  
    documents1.insert(1,base_str)
    # To make uniformed vectors, both documents need to be combined first. 
    embeddings = vectorizer.fit_transform(documents1)
    cosine_similarities = []
    cosine_similarities = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()
    
    embeddings    = []
    highest_score = 0
    highest_score_index = 0
    
    for i, score in enumerate(cosine_similarities):
        if highest_score < score:
            highest_score = score
            highest_score_index = i

    most_similar_document = documents[highest_score_index]
    
    return highest_score


# In[28]:


def FindMaxLength(lst):
    maxLength = max(len(x) for x in lst )
    return maxLength


# #### Sort the vectors in descending order of number of bounding boxes 

# In[29]:


bboxlist = copy.deepcopy(bb)
bboxlisto = copy.deepcopy(bbo)


# In[30]:


sorted_list = sorted(bboxlist, key=len, reverse=True)
bboxlist = sorted_list

dfsorted = pd.DataFrame(sorted_list)
dfsorted.columns= ["V"+str(i) for i in range(dfsorted.shape[1])]

dfsorted


# In[31]:


# sort bboxlist with actual text
sorted_listo = sorted(bboxlisto, key=len, reverse=True)
bboxlisto = sorted_listo

dfsortedo = pd.DataFrame(sorted_listo)
dfsortedo.columns= ["V"+str(i) for i in range(dfsortedo.shape[1])]

dfsortedo


# #### Rearrangement and padding of vectors where similar vectors are not present in other documents

# In[32]:


maxlen = FindMaxLength(bboxlist)
similarity = []
i =0
j =1
print('len(bboxlist[0]): ', len(bboxlist[0]))
for k in range(0,len(bboxlist[0])):
    for j in range(1,len(bboxlist)):
        i = 0
        while (i < FindMaxLength(bboxlist)):
            # finding sum of squares
            if (i >= len(bboxlist[j])):
                break
            print('base doc vector, current doc number, current doc vector',k,j,i)            
            a = bboxlist[0][k][4]
            b = bboxlist[j][i][4]

            if a != b:
                similarity1 = 0
            else:
                similarity1 = 1
            if (i < k):
                similarity1 = 0
            similarity.append(similarity1)
            similarity1 = []
            i = i+1
        print('similarity',similarity)
        maxindex = similarity.index(max(similarity))
        print('maxindex',maxindex)

        if max(similarity) == 0:
            bboxlist[j].insert(k, [0, 0, 0, 0,[]])
            bboxlisto[j].insert(k, [0, 0, 0, 0,'null'])
            
        similarity = []
        similarity1= []
        
        if maxindex == 0:
            continue
        print('***maxindex, k : ', maxindex, k)
        
        if maxindex != k and k < len(bboxlist[j]):
            save = bboxlist[j][k]
            bboxlist[j][k] = bboxlist[j][maxindex]
            bboxlist[j][maxindex] = save

            save1= bboxlisto[j][k]
            bboxlisto[j][k] = bboxlisto[j][maxindex]
            bboxlisto[j][maxindex] = save1
           
        maxindex == 0
    similarity = []
    similarity1= []
    maxindex   = []


# In[33]:


bboxcopy = copy.deepcopy(bboxlist)
bboxcopyo= copy.deepcopy(bboxlisto)


# In[34]:


df = pd.DataFrame(bboxcopy)
df.columns= ["V"+str(i) for i in range(df.shape[1])]

df


# In[35]:


df = pd.DataFrame(bboxcopyo)
df.columns= ["V"+str(i) for i in range(df.shape[1])]

df


# #### Normalization - Reshaping all the bounding boxes with corresponding box of maximum shape

# In[36]:


vectorlen = len(bboxcopy[0])
listlen   = len(bboxcopy)


# In[37]:


arealist = []

for j in range(0,vectorlen):
    for k in range(0,listlen):
        area = bboxcopy[k][j][2]*bboxcopy[k][j][3]
        arealist.append(area)
        print(arealist)
    maxarea = max(arealist)
    maxindex= arealist.index(maxarea)
    arealist = []
    
    print('max values',bboxcopy[maxindex][j][2], bboxcopy[maxindex][j][3])
    
    for k in range(0,listlen):
        bboxcopy[k][j][2] = bboxcopy[maxindex][j][2]
        bboxcopy[k][j][3] = bboxcopy[maxindex][j][3]
        
        bboxcopyo[k][j][2] = bboxcopyo[maxindex][j][2]
        bboxcopyo[k][j][3] = bboxcopyo[maxindex][j][3]


# In[38]:


### After normalization
df = pd.DataFrame(bboxcopy)
df.columns= ["V"+str(i) for i in range(df.shape[1])]

df.head()


# In[39]:


## bboxlist with actual content after normalization
dfo = pd.DataFrame(bboxcopyo)
dfo.columns= ["V"+str(i) for i in range(dfo.shape[1])]

dfo.head()


# #### Extracting only bounding box dimensions from the vector after rearrangement for structural similarity

# In[40]:


vectorlist, bboxlist2 = [], []
for k in range (0,len(bboxcopy)):
    for j in range (0,len(bboxcopy[k])):
        vector = bboxcopy[k][j][0:4]
        vectorlist.append(vector)
    bboxlist2.append(vectorlist)
    vectorlist = []

bboxlist2


# In[41]:


### After normalization
df2 = pd.DataFrame(bboxlist2)
df2.columns= ["V"+str(i) for i in range(df2.shape[1])]

df2


# #### Apply K Means clustering after reshaping vectors & normalization

# In[42]:


colors = 10*["r","b","c","g","y"]
class K_Means:
    def __init__(self, k, tol, max_iter, dist):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.dist = dist
    
    def fit(self,data):
        self.centroids = {}
        self.dist = 10
        for i in range(self.k):
            self.centroids[i] = data[i+1]
            print("***centroid#",i,":",self.centroids[i])

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                print('featureset',featureset)
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                print('distances from centroids',distances)
                classification = distances.index(min(distances))
                print('###########classification',classification)
                self.classifications[classification].append(featureset)
#             print('%%%%%%%%self.classifications',self.classifications)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
                          
            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False
            if optimized:
                dist = 0
                for i in range(k):
                    size = len(self.classifications[i])
                    featureset1 = [None] * size
                    for j in range(size):
                        featureset1[j] = self.classifications[i][j]
                        dist = dist + (np.linalg.norm(featureset1[j] -self.centroids[i]))
                    self.dist = dist
                break
                  
    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    
def clusterplot(k):        
    for classification in clf.classifications:
        color = colors[classification]
        for featureset in clf.classifications[classification]:  
            plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

    for centroid in clf.centroids:
        plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color='black', s=150, linewidths=5)

    plt.show()


# In[43]:


X_train = np.array(bboxlist2)
k     = 2
tol   = 0.001
iters = 500
clf  = K_Means(k,tol,iters,0)
clf.fit(X_train)


# In[44]:


cluster = []
for featureset in X_train:
    predict = clf.predict(featureset)
    cluster.append(predict)
labels = cluster
labels


# In[84]:


# Silhoutte Score

score = silhouette_score(X1, labels, metric='euclidean')
print('Silhouette Score: %.3f' % score)


# #### Visualization after bringing all the documents to 1 structure

# In[85]:


imgdata = []
import numpy as np
img = np.zeros([3500,2500,3],dtype=np.uint8)
img.fill(255) 
h,h2,y = 0,0,0
j=16

label = []
label = ['ACHIEVEMENTS','EDUCATION DETAILS','PERSONAL DETAILS','PROFESSIONAL DETAILS']

for i in range(0,len(bboxcopyo[j])):
    x = 30
    y = h2+40
    w = bboxcopyo[j][i][2]
    h = bboxcopyo[j][i][3]
    h2 = h2 + h +40

    if h2 > 3500:
        h2 = 0
        y = 40
        imgdata.append(img)
        img = np.zeros([3500,2500,3],dtype=np.uint8)
        img.fill(255) 
        text = bboxcopyo[j][i][4]
        rect = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, label[i] , (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80,80,80), 2)
        h2 = h + 40
    else:
        text = bboxcopyo[j][i][4]
        rect = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, label[i] , (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80,80,80), 2)
        
    k  = 0
    l  = 40         
    w1 = int(w/18)
    
    while k < len(text):
        if len(text) > w1:
            cv2.putText(img,text[k:k+w1],(x+10,y+l),0,1,(0, 0, 0))
            k = k+w1
            l = l+40
        else:              
            cv2.putText(img,text,(x+10,y+60),0,1,(0, 0, 0))
            k = len(text)

imgdata.append(img)

for i in range(0,len(imgdata)):
    fig = plt.subplots(figsize =(20, 20))
    plt.imshow(imgdata[i])  
    plt.show()


# #### Code for FastAPI endpoint

# In[51]:


import pickle
#save the model to disk
pickle.dump(clf, open('Kmclustering.pkl', 'wb'))
#load the model from disk
loaded_model = pickle.load(open('Kmclustering.pkl', 'rb'))
predict = clf.predict(featureset)

result = loaded_model.predict(featureset)

print(result)


# In[52]:


for i in range(0,len(imgdata)):
    # other things you need to do snipped
    cv2.imwrite(f'C:/Users/SarasPraveen/OneDrive/Documents/Sara AIML/image_{i}.jpg',imgdata[i])


# #### Measure Section wise content similarity 

# In[53]:


### Consider only the content from all the vectors to measure section wise content similarity

vectorlisto, bboxlisto2 = [], []
for k in range (0,len(bboxcopyo)):
    for j in range (0,len(bboxcopyo[k])):
        vector = bboxcopyo[k][j][4]
        vectorlisto.append(vector)
    bboxlisto2.append(vectorlisto)
    vectorlisto = []


# In[54]:


dfo2 = pd.DataFrame(bboxlisto2)
dfo2.columns= ["V"+str(i) for i in range(dfo2.shape[1])]

dfo2.head()


# In[55]:


dfo2 = dfo2.rename(columns={'V0':'Achievements','V1':'Education Details','V2':'Personal Details','V3':'Professional Details'})


# In[56]:


dfo2.to_json('file1.json', orient = 'split', compression = 'infer', index = 'true')


# In[57]:


# # reading the JSON file
# dfj = pd.read_json('file1.json', orient ='split', compression = 'infer')
 
# # displaying the DataFrame
# dfj.head()


# #### Pre-process the content to measure section wise content similarity

# In[58]:


for k in range(0,len(bboxlisto2)):
    for j in range(0,len(bboxlisto2[k])):    
        text = preprocess_text(bboxlisto2[k][j])
        bboxlisto2[k][j] = text


# In[59]:


dfnew = pd.DataFrame(bboxlisto2)
dfnew.columns= ["V"+str(i) for i in range(dfnew.shape[1])]

dfnew


# #### K Means algorithm for clustering based on section wise content similarity

# In[60]:


colors = 10*["r","b","c","g","y"]
class K_Means1:
    def __init__(self, k, tol, max_iter, dist):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.dist = dist
    
    def fit(self,data):
        self.centroids = {}
        self.dist = 10
        for i in range(self.k):
            self.centroids[i] = data[i+1]
            print("***centroid#",i,":",self.centroids[i])

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []
            similarity1 = []
            for featureset in data:
                print('############featureset#############',featureset)
                print(len(featureset))
                similarity = [process_tfidf_similarity(featureset,self.centroids[centroid]) for centroid in self.centroids]
                print('******distances from centroids******',similarity)
                classification = similarity.index(max(similarity))
                print('######classification######', classification)
                self.classifications[classification].append(featureset)
                similarity1.append(similarity)
            print('similarity1',similarity1)
#             print('%%%%%%%%%%self.classifications',self.classifications)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                print('---------PRINT SELF-------',classification, self.classifications[classification])
                    
    def predict(self,data):
        similarity = [process_tfidf_similarity(featureset,self.centroids[centroid]) for centroid in self.centroids]
        classification = similarity.index(max(similarity))
        return classification
    
def clusterplot(k):
         
    for classification in clf.classifications:
        color = colors[classification]
        for featureset in clf.classifications[classification]:  
            plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

    for centroid in clf.centroids:
        plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color='black', s=150, linewidths=5)

    plt.show()


# In[61]:


X_traino = np.array(bboxlisto2)
k     = 2
tol   = 0.01
iters = 100
clf2   = K_Means1(k,tol,iters,0)
clf2.fit(X_traino)


# In[89]:


cluster = []
for featureset in X_traino:
    predict = clf2.predict(featureset)
    cluster.append(predict)
cluster

labels = cluster
labels


# In[92]:


# Silhoutte Score

score = silhouette_score(X1, labels, metric='euclidean')
print('Silhouette Score: %.3f' % score)


# #### Consider only vector 1 for measuring section wise content similarity & clustering based on this

# In[64]:


dfo2 = dfo2.rename(columns={'Achievements':'V0','Education Details':'V1','Personal Details':'V2','Professional Details':'V3'})
dfo2


# In[65]:


vectorizer = TfidfVectorizer()
corpus1= dfo2['V3'].tolist()
X1 = vectorizer.fit_transform(corpus1)


# #### Elbow method to identify the optimal number of clusters

# In[66]:


corpus2= dfo2['V0'].tolist()
X2 = vectorizer.fit_transform(corpus2)


# In[67]:


from sklearn.cluster import KMeans

Sum_of_squared_distances = []
K = range(2,6)
for k in K:
   km = KMeans(n_clusters=k, max_iter=200, n_init=10)
   km = km.fit(X2)
   Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# #### Using KMeans for section wise content similarity

# In[68]:


true_k = 3
km = KMeans(n_clusters=true_k, max_iter=200, n_init=10)
km = km.fit(X1)
labels = km.labels_
labels


# In[69]:


# Silhoutte Score

score = silhouette_score(X1, km.labels_, metric='euclidean')
print('Silhouette Score: %.3f' % score)


# #### Using KMeans++ for section wise content similarity

# In[71]:


true_k = 3
km1 = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
km1 = km1.fit(X1)
labels = km1.labels_
labels


# In[73]:


km1.predict(X1)


# In[74]:


# Silhoutte Score

score = silhouette_score(X1, km1.labels_, metric='euclidean')
print('Silhouette Score: %.3f' % score)


# #### Wordcloud for the 3 clusters formed

# In[75]:


from wordcloud import WordCloud
result={'cluster':labels,'wiki':corpus1}
result=pd.DataFrame(result)
for k in range(0,true_k):
    s=result[result.cluster==k]
    text=s['wiki'].str.cat(sep=' ')
    text=text.lower()
    text=' '.join([word for word in text.split()])
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    print('Cluster: {}'.format(k))

    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# #### Content similarity with 1 vector using texthero for pre-processing the text

# In[76]:


dfo2.head(2)


# In[77]:


dfo2['clean_text_V0'] = hero.clean(dfo2['V0'])
dfo2['clean_text_V1'] = hero.clean(dfo2['V1'])
dfo2['clean_text_V2'] = hero.clean(dfo2['V2'])
dfo2['clean_text_V3'] = hero.clean(dfo2['V3'])


# In[78]:


dfo2['tfidf_V0'] = (hero.tfidf(dfo2['clean_text_V0'], max_features=3000))
dfo2['tfidf_V1'] = (hero.tfidf(dfo2['clean_text_V1'], max_features=3000))
dfo2['tfidf_V2'] = (hero.tfidf(dfo2['clean_text_V2'], max_features=3000))
dfo2['tfidf_V3'] = (hero.tfidf(dfo2['clean_text_V3'], max_features=3000))


# In[79]:


X_list = dfo2['tfidf_V3'].values.tolist()


# In[80]:


true_k = 3
clf3 = KMeans(n_clusters=true_k, max_iter=200, n_init=10)
clf3.fit(X_list)
labels=clf3.labels_
labels


# In[81]:


# Silhoutte Score

score = silhouette_score(X_list, labels, metric='euclidean')
print('Silhouette Score: %.3f' % score)


# ## CONCLUSION:
# 
# ##### Evaluation Metrics:
# 
#     Silhoutte Score for Structural similarity : 0.468
#     
#     Silhoutte Score for Content similarity of all sections: 0.319
# 
#     Silhoutte Score for section wise content similarity using custom text pre-processing : 0.485
# 
#     Silhoutte Score for section wise content similarity using texthero for text cleaning : 0.511
#     
# ##### Hence, Structure Aware Document Clustering is considered to be a better way of clustering the resume documents especially when significant sections are considered for measuring similarity.
