import docx2txt

# Passing docx file to process function
text = docx2txt.process("test.docx")

# Saving content inside docx file into output.txt file
with open("output.txt", "w") as text_file:
	print(text, file=text_file)