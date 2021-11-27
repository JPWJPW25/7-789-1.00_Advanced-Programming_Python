import urllib.request
from bs4 import BeautifulSoup
  
# here we have to pass url and path
# (where you want to save ur text file)
urllib.request.urlretrieve("https://www.deutsche-startups.de/2021/11/25/7-climatetech-startups-solarize/",
                           "C:/Users/jan-p/OneDrive/Desktop/MBF/Course Work/03_HS22/MBF_Advanced programming/Plagiarism detector/text_file2.txt")
  
file = open("text_file2.txt", "r", encoding="utf-8")
contents = file.read()
soup = BeautifulSoup(contents, 'html.parser')
  
f = open("test2.txt", "w", encoding="utf-8")
  
# traverse paragraphs from soup
for data in soup.find_all("p"):
    sum = data.get_text()
    f.writelines(sum)
  
f.close()