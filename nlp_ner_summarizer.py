import streamlit as st
import spacy 
import spacy_streamlit
from spacy import displacy
from gensim.summarization import summarize
from urllib.request import urlopen
from bs4 import BeautifulSoup 

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
nlp = spacy.load('en_core_web_sm')

def nam_ent(text):
    return nlp(text)

def url_analyses(text):
    page = urlopen(text)
    soup = BeautifulSoup(page)
    extract_info = ' '. join(map(lambda p: p.text, soup.find_all('p')))
    return extract_info




def main():
    st.title('Text Summarization and Named Entity Recognition')
    choices = ['Summarization','NER', 'Summarize and NER on urls']
    menu = st.sidebar.selectbox('Action Center',choices)
    
    if menu == 'Summarization':
        st.subheader('Summarzing blocks')
        raw_text = st.text_area('Enter text here','Type here')
        if st.button('Summarize'):
            summ = summarize(raw_text)
            st.write(summ)
    
    if menu == 'NER':
        st.subheader('Named Entity Recognition on Textual context')
        raw_text = st.text_area('Enter text here', 'Type here')
        if st.button('Find Entity'):
            result = nam_ent(raw_text)
            result = spacy_streamlit.visualize_ner(result, labels=nlp.get_pipe("ner").labels, show_table = False)
            st.write(result)

    if menu == 'Summarize and NER on urls':
        st.subheader('NLP based summarization in specified URL')
        raw_text = st.text_area('Enter path', 'Type here')
        text_length = st.slider('Length of textual-Information',50,100)
        if st.button('Extract information'):
            if raw_text != 'Type here':
                result = url_analyses(raw_text)
                len_total = len(result)
                collective_len = round(len_total/text_length)
                st.write(result[:collective_len])
                summary = summarize(result)
                summary_result = nam_ent(summary)
                summary_result = spacy_streamlit.visualize_ner(summary_result, labels=nlp.get_pipe("ner").labels, show_table = True)
                st.write(summary_result)

if __name__ == '__main__':
    main()