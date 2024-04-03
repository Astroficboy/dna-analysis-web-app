import streamlit as st
from io import StringIO
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import time


sample_data = pd.read_table('human_data.txt')
st.subheader('This is how the data should look.')
st.write(sample_data.head())
#file uploader
file = st.file_uploader('Upload .txt or .csv', type=['txt', 'csv'])
data = pd.DataFrame(columns=['sequence'])
if file is not None:
    data = pd.read_table(file)
    # st.write(data.head())
else:
    st.write("Something went wrong, try uploading again.")
class_dict={0:'G Protein couple receptors(GPCR)',
            1:'Tyrosine kinase',
            2:'Tyrosine phosphatase',
            3:'Synthetase',
            4:'Synthase',
            5:'Ion Channel',
            6:'Transcription factor'}
data['class_name'] = data['class'].map(class_dict).fillna('Not Identified')
st.bar_chart(data['class_name'].value_counts())
#writing a report
st.write('''

1. **G Protein-Coupled Receptors (GPCRs):**
   - **Increased Activity:** 
     - Hyperactivation of GPCRs can lead to overstimulation of downstream signaling pathways, causing excessive cellular responses.
     - This can result in conditions such as hypertension, cardiac arrhythmias, and hyperthyroidism.
   - **Decreased Activity:** 
     - Reduced GPCR activity may result in diminished cellular responses to extracellular signals, leading to impaired physiological processes.
     - It can contribute to conditions such as hypotension, bradycardia, and hormone insensitivity.

2. **Tyrosine Kinase Receptors:**
   - **Increased Activity:**
     - Hyperactivation of tyrosine kinase receptors can promote excessive cell growth, proliferation, and survival, leading to cancer development.
     - It can also contribute to pathological conditions such as autoimmune diseases and metabolic disorders.
   - **Decreased Activity:**
     - Reduced tyrosine kinase receptor activity may impair cell growth, differentiation, and survival, leading to developmental abnormalities or cell death.
     - It can contribute to disorders such as growth retardation, infertility, and neurodegenerative diseases.

3. **Tyrosine Phosphatases:**
   - **Increased Activity:**
     - Overexpression or hyperactivation of tyrosine phosphatases may lead to excessive dephosphorylation of target proteins, disrupting cellular signaling pathways.
     - It can result in conditions such as insulin resistance, immune dysfunction, and neurological disorders.
   - **Decreased Activity:**
     - Reduced tyrosine phosphatase activity may result in aberrant protein phosphorylation and dysregulated signaling cascades, contributing to various diseases.
     - It can lead to conditions such as cancer, autoimmune disorders, and neurodevelopmental abnormalities.

4. **Synthetase and Synthase:**
   - **Increased Activity:**
     - Upregulation of synthetase or synthase activity may lead to excessive production of macromolecules or small molecules, disrupting cellular homeostasis.
     - It can cause metabolic imbalances, accumulation of toxic intermediates, and cell stress.
   - **Decreased Activity:**
     - Downregulation or loss of synthetase or synthase activity may impair the synthesis of essential molecules, leading to deficiencies in cellular processes.
     - It can result in conditions such as metabolic disorders, impaired growth, and organ dysfunction.

5. **Ion Channels:**
   - **Increased Activity:**
     - Hyperactivation of ion channels can lead to abnormal ion fluxes, membrane depolarization, and cellular excitability, contributing to pathological conditions such as epilepsy, cardiac arrhythmias, and muscle spasms.
   - **Decreased Activity:**
     - Reduced ion channel activity may disrupt ion homeostasis, impairing cellular excitability, neurotransmission, and muscle contraction.
     - It can lead to conditions such as paralysis, sensory deficits, and cardiac conduction abnormalities.

6. **Transcription Factors:**
   - **Increased Activity:**
     - Hyperactivation of transcription factors may lead to dysregulated gene expression, promoting cell proliferation, inflammation, and tissue remodeling.
     - It can contribute to cancer progression, autoimmune diseases, and fibrosis.
   - **Decreased Activity:**
     - Reduced transcription factor activity may impair gene expression programs required for normal development, differentiation, and tissue homeostasis.
     - It can result in developmental defects, immune dysfunction, and metabolic disorders.

''')


#getting K-mers
def getKmers(sequence, size=10):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size +1)]
data['words'] = data.apply(lambda x: getKmers(x['sequence']), axis=1)
data = data.drop('sequence', axis=1)
data_text = list(data['words'])
for item in range(len(data_text)):
    data_text[item] = ' '.join(data_text[item])
label_data = data.iloc[:, 0].values
st.write(data)


#bag of words
cv = CountVectorizer(ngram_range=(4,4))
X = cv.fit_transform(data_text)


#model training
x_train, x_test, y_train, y_test = train_test_split(X, label_data, test_size=0.2, random_state=42)
classifier = MultinomialNB(alpha=0.1)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)


#metrics
def get_metrics(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
#     st.write(matrix)

# get_metrics(y_test, y_pred)


