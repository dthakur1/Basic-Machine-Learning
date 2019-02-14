# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 21:36:08 2017

@author: dushy
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#from __future__ import print_function
import os
import sys
import math
import re
from collections import Counter
#from nltk.stem import *
#from nltk.corpus import stopwords

        
#stemmer = PorterStemmer()



def classifyTrainDoc(Docname,Voc,DicSPAM,DicHAM):
    words = []
    with open(Docname, 'r') as foo:
        temp = foo.read().split()
        for x in temp:
            x = re.sub('\W+','',x)
#            x = stemmer.stem(x)     ##Add this to make it more accurate
            if x in Voc:
                words.append(x)
    PinSpam = 122/462   # Probability of class Spam
    PinHam = 340/462    # Probability of class Ham
    
    for word in words:
        PinSpam += DicSPAM[word]
        PinHam += DicHAM[word]
    return PinSpam>PinHam               ##Return if its a spam of not   

"""

a = "./train/ham/"
b = "./train/spam/"

filesystem = [a,b]       #  directory name 
abcd = []               #creatig a big list of lists of all the words in every file in every directory
for x in filesystem:    
    words = []          #words inside each file
    numOfDocs = 0   
    for y in os.listdir(x):     #list of files inside the directory
        numOfDocs += 1
        with open(x+y, 'r') as foo:   
            words += foo.read().split() #Add all the words in the file to the list
            abcd.append(words)
    abcd.append((words,numOfDocs))   #Each tuple in this big list contains all the words in the training class and the number of instances(docs) of that class


LTrH, LTrS = abcd[0][0], abcd[1][0]     ###### Need LTeS and LTeH as a list of lists
docsTrH, docsTrS = abcd[0][1], abcd[1][1]



totalDocs = docsTrH + docsTrS 
P_S = math.log(docsTrS/totalDocs)
P_H = math.log(docsTrH/totalDocs)

for i in range(len(LTrH)):
    LTrH[i] = re.sub('\W+','', LTrH[i])     ## Stripping the special characters from the words
for i in range(len(LTrS)):
    LTrS[i] = re.sub('\W+','', LTrS[i])

LTrH = filter(lambda i: not str.isdigit(i), LTrH)  # Stripping the numbers from the words
LTrS = filter(lambda i: not str.isdigit(i), LTrS)

LTrH = [x for x in LTrH if x != '']
LTrS = [x for x in LTrS if x != '']


"""



############## This part is commented out because the projet folder contains the stemmed word files and remote does not have NLTK imported



#NEWLTrH = [stemmer.stem(word) for word in LTrH]     # Retreiving the stem of the words and updating the list with these stems
#NEWLTrS = [stemmer.stem(word) for word in LTrS]     # Stemmer also takes care of making the words lowercase.

#NewWOStopHam = [word for word in NEWLTrH if word not in (stopwords.words('english'))]       # Making another list from the words in ham and spam directories but removing the stopwords
#NewWOStopSpam = [word for word in NEWLTrS if word not in (stopwords.words('english'))]




##############Making text file and putting the stemmed words of both the directories(Ham and Spam) with both the cases(with and without stopwords)

############## This part is commented out because the projet folder contains these files and remote does not have NLTK imported
#with open('StemmedTrainSpam.txt', 'w') as fp:
#    for i in NEWLTrS:
#        fp.write("%s\n"%i)


#with open('StemmedTrainHam.txt', 'w') as fp:
#    for i in NEWLTrH:
#        fp.write("%s\n"%i)


#with open('StemmedTrainSpam_WITHOUT_Stop.txt', 'w') as fp:
#    for i in NewWOStopSpam:
#       fp.write("%s\n"%i)


#with open('StemmedTrainHam_WITHOUT_Stop.txt', 'w') as fp:
#    for i in NewWOStopHam:
#        fp.write("%s\n"%i)
        
        
########################################################################################## New lists which will be updated from the stemmed words file

        
StemmedListTrainHam = []
StemmedListTrainSpam = []        

StemmedListTrainHamSTOP = []
StemmedListTrainSpamSTOP = []        



######### Reading the words from the file of stemmed words of Ham and Spam directories.

with open('StemmedTrainSpam.txt','r') as foo:                   
            StemmedListTrainSpam+=(foo.read().split())

with open('StemmedTrainHam.txt','r') as foo:
            StemmedListTrainHam+=(foo.read().split())            



with open('StemmedTrainSpam_WITHOUT_Stop.txt','r') as foo:
            StemmedListTrainSpamSTOP += (foo.read().split())

with open('StemmedTrainHam_WITHOUT_Stop.txt','r') as foo:
            StemmedListTrainHamSTOP += (foo.read().split())            

############### Vocab and Vocab2 are the vocabulary with and without stop words.
Vocab = list(set(StemmedListTrainHam+StemmedListTrainSpam))
Vocab2 = list(set(StemmedListTrainHamSTOP+StemmedListTrainSpamSTOP))

############### DTrH contains the counts of each unique word in the list of all Ham words. Same nomenclature for other dictionaries
DTrH = dict(Counter(StemmedListTrainHam))
DTrS = dict(Counter(StemmedListTrainSpam))

DTrH2 = dict(Counter(StemmedListTrainHamSTOP))
DTrS2 = dict(Counter(StemmedListTrainSpamSTOP))



##############  This is the list of words present in one class but not in other. This will is used for smoothening.
NotInHam = list(set(Vocab) - set(DTrH.keys()))
NotInSpam = list(set(Vocab) - set(DTrS.keys()))


NotInHamSTOP = list(set(Vocab2) - set(DTrH2.keys()))
NotInSpamSTOP = list(set(Vocab2) - set(DTrS2.keys()))



############  Add the words to the dictionary of words and set their count to zero

for word in NotInHam:
    DTrH.update({word:0})
for word in NotInSpam:
    DTrS.update({word:0})

for word in NotInHamSTOP:
    DTrH2.update({word:0})
for word in NotInSpamSTOP:
    DTrS2.update({word:0})

########## total number of words in each case
totalInHam = sum(DTrH.values())+len(DTrH)
totalInSpam = sum(DTrS.values())+len(DTrS)

totalInHamSTOP = sum(DTrH2.values())+len(DTrH2)
totalInSpamSTOP = sum(DTrS2.values())+len(DTrS2)



########## Laplace 1 smoothening. Now the dictionaries contain conditional probabilities.
for word in DTrH.keys():
    DTrH[word]+=1
    DTrH[word] = math.log((DTrH[word])/totalInHam)
for word in DTrS.keys():
    DTrS[word]+=1
    DTrS[word] = math.log((DTrS[word])/totalInSpam)


for word in DTrH2.keys():
    DTrH2[word]+=1
    DTrH2[word] = math.log((DTrH2[word])/totalInHamSTOP)
for word in DTrS2.keys():
    DTrS2[word]+=1
    DTrS2[word] = math.log((DTrS2[word])/totalInSpamSTOP)

############################################################################# Testing in spam folder of Test directory
############################################################################# 



AccInSpam = 0
AccInSpamSTOP = 0

DocsInSpam = 0

for y in os.listdir('test/spam'):
    DocsInSpam+=1
    if classifyTrainDoc('test/spam/'+y,Vocab, DTrS, DTrH):
        AccInSpam+=1

for y in os.listdir('test/spam/'):
    if classifyTrainDoc('test/spam/'+y,Vocab2, DTrS2, DTrH2):
        AccInSpamSTOP+=1


        
AccInHam = 0
AccInHamSTOP = 0

DocsInHam = 0

for y in os.listdir('test/ham/'):
    DocsInHam+=1
    if not classifyTrainDoc('test/ham/'+y,Vocab, DTrS, DTrH):
        AccInHam+=1

for y in os.listdir('test/ham'):
    if not classifyTrainDoc('test/ham/'+y,Vocab2, DTrS2, DTrH2):
        AccInHamSTOP+=1
        

AccInSpam = AccInSpam/DocsInSpam    
AccInHam = AccInHam/DocsInHam      

AccInSpamSTOP = AccInSpamSTOP/DocsInSpam    
AccInHamSTOP = AccInHamSTOP/DocsInHam        
  




with open('results.txt', 'w') as fp:
        fp.write("%f\t%f\n%f\t%f"%(AccInSpam,AccInSpamSTOP,AccInHam,AccInHamSTOP))
    
fp.close()    
                
    












    
    
    

