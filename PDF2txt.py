import os
import PyPDF2

os.chdir('C:/Users/jan-p/OneDrive/Desktop/Text2Pdf Task')
os.getcwd()

pdf2convert = open("Allianz_Group_Sustainability_Report_2020-web.pdf", "rb")

m = "Allianz"

pdfreader = PyPDF2.PdfFileReader(pdf2convert)

y = pdfreader.getNumPages()

for n in range(0,y):
    pageobj = pdfreader.getPage(n)
    text = pageobj.extractText()
    file1=open(r"C:/Users/jan-p/OneDrive/Desktop/Text2Pdf Task/text_" + m + ".txt","a")
    file1.writelines(str(text.encode('utf8')))



