from cmath import isnan
import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import CountVectorizer

#"""Pull in Keywords which similarity will be based upon"""
keywords = pd.read_csv("keyword dataset v1.xlsx - Sheet1.csv", header=None)
keyword_list = keywords[0].to_list() + keywords[1].to_list() + keywords[2].to_list() + keywords[3].to_list() + keywords[4].to_list()
keyword_list = [i for i in keyword_list if str(i) !='nan']

#"""## Creating a Vectorizer Object

#fit vectorizer on the keywords
#"""


# Create a Vectorizer Object
vectorizer = CountVectorizer()
vectorizer.fit(keyword_list)

def similarity_score(job_description_a, job_description_b):
  jda = vectorizer.transform([job_description_a])
  A = jda.toarray()[0]
  jdb = vectorizer.transform([job_description_b])
  B = jdb.toarray()[0]
  similarity = np.dot(A,B)/(norm(A)*norm(B))
  if np.isnan(similarity):
    return(0)
  else:
    return(similarity.round(4)*100)


def main():
    
    # ===================== Set page config and background =======================
    # Main panel setup
    # Set website details
    st.set_page_config(page_title ="Job Description Similarity Scorer", 
                       page_icon=':desktop_computer:', 
                       layout='centered')
    """## Job description Similarity Scorer"""

    with st.expander("About"):
        st.write("This App checks for the similarity between two job descriptions and returns the score, the model uses certain keywords which can be found [here](https://docs.google.com/spreadsheets/d/1ILSICeRE3GQRiQGbRfoChAoP6PZSQt6f)")

    with st.form(key = 'form1', clear_on_submit=False):
        Job_description1 = st.text_area("First Job description")
        Job_description2 = st.text_area("Second Job description")
        submit_button = st.form_submit_button()

    if submit_button:
        st.success("I'm processing your request")
        st.write("The Similarity score between the two Job descriptions is {}%".format(similarity_score(Job_description1,Job_description2)))

if __name__ == "__main__":
    main()