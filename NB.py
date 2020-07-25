# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:42:27 2019

@author: cricketjanoon
"""

import numpy as np
import re
file = open("LSI-data.txt")

#------------------------
numberOfTrainingSet = 32604
#------------------------

# testDoc = "hp raises dividend, touts cloud computing"
# testDoc = "china hails north korean leader as he tours country"
#testDoc = "jack warner's threatened soccer tsunami remains stuck in the doldrums, the corruption"
#testDoc = "strip mall to catch a glimpse of the boxer manny pacquiao working"
# testDoc = "andrew d. hamingson, point to anger about costly"
testDoc = "government resigns: state television"
testDoc = re.sub(r'[^\w\s_]+', '', testDoc)


# 0 business
# 1 entertainment
# 2 health
# 3 sci_tech
# 4 sport
# 5 us
# 6 world
classes = ["business", "entertainment", "health", "sci_tech", "sport", "us", "world"]

#making data matrix
documentMatrix = np.full((numberOfTrainingSet,3), dtype=object, fill_value = "empty")
documentCount = 0
index = 0
for line in file:
	if documentCount >= numberOfTrainingSet:
		break
	
	if line in ['\n','\r\n']:
		documentCount = documentCount + 1
		index = 0
	else:
		index = index + 1
		
	if index == 1:
		documentMatrix[documentCount,1] = line
	elif index == 2:
		documentMatrix[documentCount,2] = line
	elif index == 7:
		documentMatrix[documentCount,0] = line

#cleaning document string of any non-alphabet characters
for i in range(documentMatrix.shape[0]):
	for j in range(documentMatrix.shape[1]):
		documentMatrix[i,j] = re.sub(r'[^\w\s_]+', '', documentMatrix[i,j]).strip()
		documentMatrix[i,j] = documentMatrix[i,j].lower()			
documentMatrix = documentMatrix.astype(str)

documentMatrix2 = np.full((numberOfTrainingSet,2), dtype=object, fill_value = "empty")
for i in range(len(documentMatrix)):
	documentMatrix2[i, 0] = documentMatrix[i,0]
	documentMatrix2[i, 1] = documentMatrix[i,1] + " " + documentMatrix[i, 2]	
documentMatrix2 = documentMatrix2.astype(str)

#docMatrix = np.load("docMatrix2.npy")
docMatrix = documentMatrix2
	
#collecting all the possible words in the test data
vocab = []
for i in range(docMatrix.shape[0]):
	for word in docMatrix[i,1].split():
		if word not in vocab:
			vocab.append(word)
#counting occurence of each class of document	
countOfEachClass = np.zeros((7))
textOfClass = ["", "", "", "", "", "", ""]
for i in range(docMatrix.shape[0]):
	if docMatrix[i,0] == "business":
		textOfClass[0] = textOfClass[0] + " " + docMatrix[i,1]
		countOfEachClass[0] = countOfEachClass[0] + 1
	if docMatrix[i,0] == "entertainment":
		textOfClass[1] = textOfClass[1] + " " + docMatrix[i,1]
		countOfEachClass[1] = countOfEachClass[1] + 1
	if docMatrix[i,0] == "health":
		textOfClass[2] = textOfClass[2] + " " + docMatrix[i,1]
		countOfEachClass[2] = countOfEachClass[2] + 1
	if docMatrix[i,0] == "sci_tech":
		textOfClass[3] = textOfClass[3] + " " + docMatrix[i,1]
		countOfEachClass[3] = countOfEachClass[3] + 1
	if docMatrix[i,0] == "sport":
		textOfClass[4] = textOfClass[4] + " " + docMatrix[i,1]
		countOfEachClass[4] = countOfEachClass[4] + 1
	if docMatrix[i,0] == "us":
		textOfClass[5] = textOfClass[5] + " " + docMatrix[i,1]
		countOfEachClass[5] = countOfEachClass[5] + 1
	if docMatrix[i,0] == "world":
		textOfClass[6] = textOfClass[6] + " " + docMatrix[i,1]
		countOfEachClass[6] = countOfEachClass[6] + 1
textArr = np.array(textOfClass)
	
probabilityOfEachClass = np.divide(countOfEachClass, docMatrix.shape[0])

wordCountInEachDocClass = np.zeros((textArr.shape[0]))
for i in range(len(textArr)):
	wordCountInEachDocClass[i] = len(textArr[i].split())
	
eachWordCountInEachClass = np.zeros((textArr.shape[0], len(vocab))) # n_k

for i in range(eachWordCountInEachClass.shape[0]):
	for j in range(eachWordCountInEachClass.shape[1]):
		eachWordCountInEachClass[i,j] = textArr[i].split().count(vocab[j])
	
for i in range(eachWordCountInEachClass.shape[0]):
	for j in range(eachWordCountInEachClass.shape[1]):
		eachWordCountInEachClass[i,j] = eachWordCountInEachClass[i,j] +1

weightMatrix = np.zeros((textArr.shape[0], len(vocab)))

for i in range(eachWordCountInEachClass.shape[0]):
	for j in range(eachWordCountInEachClass.shape[1]):
		weightMatrix[i,j] = (eachWordCountInEachClass[i,j] + 1) / (wordCountInEachDocClass[i] + len(vocab))


#testing phase of naive bayes
positions = np.zeros(len(testDoc.split()), dtype="int32")

index = 0
for word in testDoc.split():
	positions[index] = vocab.index(word)
	index = index + 1
	

testProb = np.zeros(len(countOfEachClass))

for i in range(len(countOfEachClass)):
	for j in range(len(positions)):
		testProb[i] = testProb[i] + weightMatrix[i, positions[j]]	

print("Test Doc: "+ testDoc)
print("Catergory predicted: " + classes[np.argmax(testProb)])
	