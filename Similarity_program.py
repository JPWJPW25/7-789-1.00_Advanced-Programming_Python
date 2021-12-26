#Install and import required libraries

#Install libraries for similarity calculations
#pip install -U scikit-learn
#pip install nltk
#pip install pdfplumber
#pip install docx2txt
#pip install seaborn

#Import and retrieve libraries and functions used for file handling
import os
from os import listdir
from os.path import isfile, isdir

#Import and retrieve libraries used to convert different file types into txt files
import docx2txt
import pdfplumber

#Import and retrieve libraries and functions used to preprocess txt files 
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize

#Import and retrieve libraries and functions used to compute similarity between txt files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Import and retrieve libraries and functions to construct data frame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


#Convert data files from different types into txt files
#Write two functions to convert from a) PDF to txt and b) Word to txt

#Function a) to convert PDF files into txt files
def convert_PDF2txt3(folderpath_pdf, fileName_pdf):
    #Access folder that containes files to be converted
    os.chdir(folderpath_pdf)
    os.getcwd()
    
    #Create empty string to pass text from pdf to
    all_text = ""
    
    #Create filepath of PDF file
    filepath_pdf = folderpath_pdf + "/" + fileName_pdf + ".pdf"
    
    #Open pdf file and extract text to empty string
    with pdfplumber.open(filepath_pdf) as pdf:
        for page in pdf.pages:
            all_text += page.extract_text()
    #Open destination txt file before writing content
    file1 = open(fileName_pdf + "_pdf.txt","w+")    
    
    #Write content from string to empty text file
    file1.write(str(all_text.encode("utf-8")))
    file1.close()
        
        
#Function b) to convert Word files into txt files
def convert_Word2txt(folderpath_word, fileName_word):
    #Access folder that containes files to be converted
    os.chdir(folderpath_word)
    os.getcwd()
    
    # Passing docx file to process function
    text = docx2txt.process(fileName_word + ".docx")

    # Saving content inside docx file into output.txt file
    file1 = open(fileName_word + "_w.txt", "w+")
    
    #Write content from string to empty text file
    file1.write(text)
    file1.close()


# In[3]:


#Prepare files and convert format to preprocess the data
#Write two functions to first retrieve file information (filepath and -name) and second to retrieve content

#Function 1: to get file information in list
def return_ListOfFilePaths(folderPath):
    #Define empty list as return of the function
    fileInfo = []
    #Define new list for fileNames and add fileNames to the list
    listOfFileNames = []
    for fileName in listdir(folderPath):
        filePath = folderPath + "/" + fileName
        if os.path.isfile(filePath) == True and fileName.endswith('.txt'):
            listOfFileNames.append(fileName)
    
    #Define new list for filePaths and add filePaths to the list
    listOfFilePaths = []
    for fileName in listdir(folderPath):
        filePath = folderPath + "/" + fileName
        if os.path.isfile(filePath) == True and fileName.endswith('.txt'):
            listOfFilePaths.append(filePath)
    
    #Append fileNames and FilePathts to return list
    fileInfo.append(listOfFileNames)
    fileInfo.append(listOfFilePaths)
    return fileInfo


#Function 2: to get file content in dictrionary
def create_docContentDict(filePaths):
    #Define empty dictionary as return of the function
    rawContentDict = {}
    
    #Iterate through filePaths list, open files and add filepaths(key) and content (value) as key value pairs to dictionary
    for filePath in filePaths:
        with open(filePath, "r") as inputfile:
            fileContent = inputfile.read()
        #Remove line breaks left over from pdf conversion for multipage PDF files    
        fileContent_cleaned_1 = fileContent.strip()
        fileContent_cleaned_2 = fileContent_cleaned_1.replace("\\n", "")
        rawContentDict[filePath] = fileContent_cleaned_2
    return rawContentDict


# In[4]:


#Preprocess the data in order to make it useable for similarity calculations
#Write four subfunctions to 1) tokenize the content 2) remove punctuations 3) convert to lower case 
#and 4) apply stemming algorithm. Finally include all subfunctions in 5) one main preprocessing function.

#Function 1: Tokenize content (i.e. splitting text into small units) by applying the word_tokenize() function
def tokenize_Content(contentsRaw):
    tokenizedContent = word_tokenize(contentsRaw)
    return tokenizedContent

#Function 2: Remove stop words
#(i.e. Stop words are those words in natural language that have a very little meaning, such as "is", "an", "the")
#by applying the stopwords() function for english language
def remove_StopWords(tokenizedContent):
    #Define set (Sets are used to store multiple items in a single variable) with relevant stopwords
    #Set can also be extended based on individual preferences
    stop_word_set = set(nltk.corpus.stopwords.words("english"))
    
    #Remove all stop words by iterating through input and check if tokens are not represented in stop_word_set
    filteredContents_a = [word for word in tokenizedContent if word not in stop_word_set]
    return filteredContents_a

#Function 2: Remove punctuation (i.e. )
def remove_Punctuation(filteredContents_a):
    #Define set (Sets are used to store multiple items in a single variable) with relevant punctuations
    #Set can also be extended based on individual preferences
    exclude_Puncuation_set = set(string.punctuation)
    doubleSingleQuote = "\\n"
    exclude_Puncuation_set.add(doubleSingleQuote)
    #Remove all punctuations by iterating through input and check if tokens are not represented in exclude_Punctuation_set
    filteredContents_b = [word for word in filteredContents_a if word not in exclude_Puncuation_set]
    return filteredContents_b


#Function 3: Convert content to lowercase by applying the lower() function
def convert_ItemsToLower(filteredContents_b):
    #Iterate through content and convert all tokens to lowercase (i.e. Man vs. man should be the same for similarity purposes))
    filteredContents_c = [term.lower() for term in filteredContents_b]
    return filteredContents_c


#Function 4: Apply stemming algorithm to retrieve morphological variants of a root/base word
def perform_PorterStemming(filteredContents_c):
    #Assign PorterStemmer() function to variable
    ps = PorterStemmer()
    
    #Iterate through tokens, stem them and put back into list
    filteredContents_d = [ps.stem(word) for word in filteredContents_c]
    return filteredContents_d


#Function 5: Include all previous functions in one main function that performs all preprocessing steps at once
def processData(rawContents):
    cleaned_a = tokenize_Content(rawContents)
    cleaned_b = remove_StopWords(cleaned_a)
    cleaned_c = remove_Punctuation(cleaned_b)
    cleaned_d = convert_ItemsToLower(cleaned_c)
    cleaned_e = perform_PorterStemming(cleaned_d)    
    return cleaned_e


# In[5]:


#Define two functions to (1) preprocess data, perform word embedding, vectorize texts and (2) compute cosine similarity
#Function 1: Use TFIDF (term frequency–inverse document frequency) by applying TfidfVectorizer() function

def vectorize_text (rawContentDict):
    #Use our prebuild processData function as tokenizer to perform custom tokenization
    #We don't include stop words here as they have already been removed and stopwords from Scikit-learn are not tokenized
    tfidf = TfidfVectorizer(tokenizer = processData)
    #Apply function to text values in dictionary
    tfs = tfidf.fit_transform(rawContentDict.values())
    
    return tfs

#Function 2: Compute the cosine similarity based on the vectorized representations of the text by applying cosine_similarity() function 
def compute_CosinesSimilarity (tfs, fileNames):
    numFiles = len(fileNames)
    #Define empty list, that will contain sublists of value pairs
    listOfValues = []
    #Define two for loops to get all possible combinations of the different documents to compare
    for i in range(numFiles):
        #Define sublist for value pairs
        subListOfValues = []
        for n in range(numFiles):
            matrixValue = cosine_similarity(tfs[i], tfs[n])
            #Cosine similarity is returned in matrix format, so we need to retrieve the similarity score with indexing
            numValue = matrixValue[0][0]
            subListOfValues.append(numValue)
        listOfValues.append(subListOfValues)
    return listOfValues
            


# In[6]:


#MAIN PROGRAM

#####################################
# Step 1: onboarding to the program #
#####################################
print("Welcome to the document similarity checker. The program lets you compare two documents and compute their similarity. Based on the similarity you can for example identify plagiats or copies of text that just have been modified by changing word orders.\n")

############################################
# Step 2: Short description of the program #
############################################
print("In a first step, the program loads documents of the type Word and PDF. Second, after loading thedocuments, the program converts the files into text files. Third, based on the created text files,the program preprocesses the data before computing the similarity with Natural LanguageProcessing (NPL) techniques. More specifically, the program computes the similarity based on theso called cosine similarity. Fourth, the results of the similarityanalysis are loaded into dataframeand plotted using a upper triangle heatmap, that can be saved as png by the user. Lastly, the userhas the opportunity to start the program again")

#####################################################################################
# Step 3: Create a while loop to potentially restart the program after finishing it #
#####################################################################################
restartProgram = "Yes"
while restartProgram == "Yes": 
    
    ####################################################################################################################
    # Step 4: Ask user for the folderpath containing the documents and the number of documents she/he wants to compare #
    ####################################################################################################################
    folderPath = input("\nPlease input the path to the folder that contains the documents you want to compare here: \n")

    #Make sure that the folderPath exists and is entered correctly
    folderExists = os.path.isdir(folderPath)
    while folderExists == False:
        folderPath = input("\nFolder does not exist. Please input the path to the folder that contains the documents you want to compare here: \n")
        folderExists = os.path.isdir(folderPath)
    
    #Make sure that the user enters the quantity of the documents as integer
    while True:
        numberOfDocs = input("\nHow many documents do you want to compare? \n")
        try:
            numberOfDocs = int(numberOfDocs)
            break
        except:
            print("Erorr! You did not enter the quantity of documents as integer. Please do it again! ")
            continue
    
    ######################################################################################
    # Step 5: Ask for the respective file names, the file types and store them in a list #
    ######################################################################################
    listWithNames = []
    listWithFormats = []
    for i in range(numberOfDocs):
        docName = input("\nWhat is the name of document Nr." + str(i+1) + "? \n") #MAXIMUM?
        docType = input("\nWhat is the type of document Nr." + str(i+1) + "? (.docx / .pdf)\n")

        #Check if file exists, if not ask as long as the input is valid
        filePath = folderPath + "/" + docName + docType
        docExists = os.path.isfile(filePath)
        while docExists == False:
            docName = input("\nFile does not exist. What is the name of document Nr." + str(i+1) + "? \n") #MAXIMUM?
            docType = input("\nWhat is the type of document Nr." + str(i+1) + "? (.docx / .pdf)\n")
            filePath = folderPath + "/" + docName + docType
            docExists = os.path.isfile(filePath)

        #Append file name and type to list if file exists
        listWithNames.append(docName)
        listWithFormats.append(docType)

    ####################################################################
    #Step 6: Create dictionary with names and corresponding file types #
    ####################################################################
    dictWithNames = {}
    for i in range(len(listWithNames)):
        dictWithNames[listWithNames[i]] = listWithFormats[i]

    #Step 7: Load and convert the submitted files into text files
    for key, value in dictWithNames.items():
        if value == ".docx":
            word1 = convert_Word2txt(folderPath, key)
        elif value == ".pdf":
            pdf1 = convert_PDF2txt3(folderPath, key)

    #########################################################################################################
    # Step 8: Retrieve file information for text files & create dictionary with raw content from text files #
    #########################################################################################################
    fileNames, filePathList = return_ListOfFilePaths(folderPath)
    rawContentDict = create_docContentDict(filePathList)
    
    ################################################
    # Step 9: Preprocess the data and vectorize it #
    ################################################
    tfs = vectorize_text(rawContentDict)
    results = compute_CosinesSimilarity(tfs, fileNames)

    ############################################################
    # Step 10: Create dataframe and visualize results in table #   
    ############################################################
    df = pd.DataFrame(results, columns = fileNames, index = fileNames)
    
    ##############################################################
    # Step 11: Ask user if to visualize the results in dataframe #
    ##############################################################
    visualizeDataframe = input("\nDo you want to see the dataframe? (Yes/No)\n")
    if visualizeDataframe == "Yes":
        #use .style function to beautify the table of results 
        display(df.style)

    ###############################################################
    # Step 12: Plot the results in colored lower triangle heatmap #
    ###############################################################
    #Define the plot as figure and set the size of the plot and the axes
    fig, ax = plt.subplots(figsize=(10, 7))
    
    #Remove the upper part of the heatmap to delete duplicate values
    dfCleaned = df.where(np.tril(np.ones(df.shape)).astype(np.bool))
    
    #Set up heatmap with seaborn.heatmap() function, including color scheme "Reds" and annotations
    sb.heatmap(dfCleaned, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1,
                square=True, linewidths=5, cbar_kws = {"shrink": .9}, annot_kws={"size": 12})
    
    #Rotate and define size of the y/x-ticks
    plt.yticks(rotation=0, size=12)
    plt.xticks(rotation=35, size=12)
    
    #Define and add title to lower triangle heatmap
    title = 'LOWER TRIANGLE HEATMAP OF DOCUMENT SIMILARITY\n'
    plt.title(title, loc='left', fontsize=14)
    
    ###################################################################################
    # Step 13: Ask user if she / he wants to save the plot as png file in same folder #
    ###################################################################################
    #Use .savefig() function to save file to folder if user wants to
    saveFig = input("\nDo you want to save the plot as png file in your folder? (Yes/No)\n")
    if saveFig == "Yes":
        plt.savefig("LOWER TRIANGLE HEATMAP OF DOCUMENT SIMILARITY.png")
        print("Perfect, the file has been saved to your folder.")
    
    #####################################################################################################################
    # Step 14: Ask user if to show the plot, this step needs to come after saving, otherwise memory is cleared for plot #
    #####################################################################################################################
    visualizePlot = input("\nDo you want to see the plot? (Yes/No)\n")
    if visualizePlot == "Yes":
        plt.show()
    
    ##################################################################
    # Step 15: Ask user if she / he wants to start the program again #
    ##################################################################
    restartProgram = input("\nDo you want to restart the program? (Yes/No) \n")
    if restartProgram == "Yes":
        print("\nAlright, let's start over again!\n")
    elif restartProgram == "No":
        print("\nAlright, thank you very much for using the program. Goodbye!\n")

