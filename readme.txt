STRUCTURE AWARE DOCUMENT CLUSTERING USING K MEANS ALGORITHM
------------------------------------------------------------

Installations Needed:
---------------------
- !pip install docx2pdf
- !pip install tesseract
- !pip install easyocr 
- !pip install opencv-python-headless
- !pip install Pillow
- !pip install spacy

Overall Approach:
-----------------
- Convert all PDFs, DOCXs => JPGs
- Draw Bounding boxes and extract text within the bounding boxes using easyOCR
- Extract the dimensions of the bounding boxes (x,y,w,h coordinates in form of vectors)
- Text pre-processing -> removal of stop words, conversion to lowercase, tokenization, lemmatization
- Extracted labels from the text using custom dictionary
- Appended the content to the bounding box vectors
  (constructed custom dictionary for extracting labels and appended the labels to vectors)
- Rearrangement of vectors based on content similarity (used TFidf vectorizer & cosine similarity)
- Padding technique to bring similar content at one vector position and post padding 
- Normalization of bounding box dimensions to bring all bounding boxes to similar structure
- Measure section wise content similarity using TF-IDF vectorizer & cosine similarity for the sake of clustering

Clustering Performed:
---------------------
  a) Cluster resumes based on structural similarity after normalization 
      (used K Means clustering - Euclidean distance L2 norm)
  b) Cluster based on section wise content similarity 
      (used K Means clustering - TFIDF vectorizer, cosine similarity)
  c) Cluster using only 1 vector (professional experience) 
      (used K Means/K Means++ scikit library to cluster)

Conclusion:
-----------
Structure Aware Document Clustering is considered to be a better way of clustering the resume documents especially 
when significant sections are considered for measuring similarity.