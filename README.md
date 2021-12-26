# 7-789-1.00_Advanced-Programming_Python - Individual project

**Title: DOCUMENT SIMILARITY CHECKER**

**Date:** 26.12.2021

**Name:** Jan-Philipp Wittmann

**Student_id:** 16-621-120

## Program description
The program lets you compare documents and compute their similarity. You can choose the number of files you want to compare. The program is compatible with Word and PDF files.
Based on the similarity you can for example identify plagiarism or copies of text that just have been modified by changing word orders. The computed similarity 
is computed by using the ntlk and scikit-learn libraries and by applying the concept of cosine similarity.
Results are stored in a dataframe and visualized in a lower triangle heatmap.

Program: "Document_similarity_checker.py"

## Dependencies 
Please install the following libraries to make the code work
```bash
pip install scikit-learn
pip install nltk
pip install pdfplumber
pip install docx2txt
pip install seaborn
```
## Structure of the program

Part 1) Import packages and specific functions

Part 2) Define functions to convert PDF or Word files to text files

Part 3) File handling functions for text files

Part 4) Define tokenizer and respective functions for data preprocessing

Part 5) Define vectorizer and consine similarity calculator

Part 6) Main program divided into 15 steps


## Example outline

**Documents used:**


1) Doc1_FCB
    - Description: Snippet from Wikipedia article of FC Bayern Basketball team
    - Type: PDF
    - Source: https://en.wikipedia.org/wiki/FC_Bayern_Munich_(basketball)

2)	Doc2_FCBB
    - Description: Snippet from Wikipedia article of FC Bayern soccer team
    - Type: Word
    - Source: https://en.wikipedia.org/wiki/FC_Bayern_Munich

3)	Doc3_Roses
    - Description: Snippet from Wikipedia article about roses
    - Type: PDF
    - Source: https://en.wikipedia.org/wiki/Rose

4)	Doc4_Roses_2
    - Description: Snippet from Wikipedia article about roses but different paragraph
    - Type: Word
    - Source: https://en.wikipedia.org/wiki/Rose


**Result:**
Please see file "Document similarity checker_Example outline.pdf"

## Example outline
Thank you for using my program!
