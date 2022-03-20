import re
import os
import string
import glob
import json
import heapq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from collections import Counter
from math import log10 as log
from math import sqrt


def vectorise(sent_count: Counter, all_tokens):
    return [sent_count[token] for token in all_tokens]


def myVector(counter, all_tokens):
    buckets = [0] * len(all_tokens)
    for item in counter:
        buckets[int(item[0])] = item[1]
    return buckets


def tf_idf_row(freq_vector, unique_tokens, doc_freq, len_docid, vocabulary):
    val_list = list(vocabulary.values())
    return [tf_idf(tf, doc_freq[str(val_list.index(term))], len_docid) for term, tf in zip(unique_tokens, freq_vector)]


def tf_idf(term_freq, doc_freq, N):
    return log(1+term_freq) * log(N/doc_freq) if term_freq != 0 else 0


def dot(vec1, vec2):
    return sum(v1*v2 for v1, v2 in zip(vec1, vec2))


def magnitude(vec):
    return sqrt(sum(v**2 for v in vec))


def normalise(vec):
    
    mag = magnitude(vec)
    return [v/mag for v in vec]


def clean_text(all_text):
    all_text = re.sub('\\s', ' ', all_text)
    all_text = re.sub('[0-9]', ' ', all_text)
    all_text = re.sub('[ ]{2,}', ' ', all_text)

    translation_table = all_text.maketrans(
        string.punctuation, ' ' * len(string.punctuation))

    all_text = all_text.translate(translation_table)

    
    tokens = word_tokenize(all_text)

    final = []

    errors = 0

    ######
    st = PorterStemmer()
    ####
    for i in tokens:

        if len(i)<=13 and len(i) > 2:
            
                #####
                i = st.stem(i.lower())
                ####
                final.append(i.lower())

    return final



def write_file(docid, postings, vocabulary, doc_freq):
    sf = open("DocID2.json", "w", encoding="utf-8")
    json.dump(docid, sf, indent=2)
    sf.close()
    sf = open("vocabulary2.json", "w", encoding="utf-8")
    json.dump(vocabulary, sf, indent=2)
    sf.close()
    sf = open("postings2.json", "w", encoding="utf-8")
    json.dump(postings, sf, indent=2)
    sf.close()
    sf = open("docfreq2.json", "w", encoding="utf-8")
    json.dump(doc_freq, sf, indent=2)
    sf.close()


def part1():

    count = 1
    vocabulary = {}
    docid = {}
    postings = {}
    doc_freq = {}
    path = 'ueapeople'

    for filename in glob.glob(os.path.join(path, '*.html')):

        if len(docid) % 50 == 0:
            print(str(len(docid)) + " " + str(len(vocabulary)))

        f = open(filename, "r", encoding="utf-8")
        soup = BeautifulSoup(f, "html.parser")
        f.close()
        all_text = soup.get_text(strip=True, separator=u' ')
        all_text = clean_text(all_text)

        token_counts = Counter(all_text)
        postings[len(docid)] = ''
        for i, count in token_counts.items():

            if (i in vocabulary.values()):
                val_list = list(vocabulary.values())
                postings[len(docid)] += ("," +
                                         str(val_list.index(i)) + "|" + str(count))
                doc_freq[val_list.index(i)] += 1

            else:
                postings[len(docid)] += ("," +
                                         str(len(doc_freq)) + "|" + str(count))
                doc_freq[len(doc_freq)] = 1
                vocabulary[len(vocabulary)] = i

        docid[len(docid)] = filename

    write_file(docid, postings, vocabulary, doc_freq)


def part2():
    sf = open("postings2.json", "r")
    postings = json.load(sf)
    sf.close()
    sf = open("docfreq2.json", "r")
    doc_freq = json.load(sf)
    sf.close()
    sf = open("docid2.json", "r")
    docid = json.load(sf)
    sf.close()

    for i, values in postings.items():
        mine = values.split(",")
        new = []
        for items in mine:
            if items != '':
                test = items.split("|")
                doc_freqs = doc_freq[str(test[0])]

                N = len(docid)
                test[1] = log(1 + int(test[1])) * \
                    log(N / doc_freqs) if int(test[1]) != 0 else 0

                new.append(test)
        val_list = list(doc_freq.keys())
        postings[i] = new
    sf = open("tf_idf2.json", "w", encoding="utf-8")
    json.dump(postings, sf, indent=2)
    sf.close()



def part4():
    sf = open("stemming/docfreq2.json", "r")
    doc_freq = json.load(sf)
    sf.close()
    sf = open("stemming/tf_idf2.json", "r")
    postings = json.load(sf)
    sf.close()
    sf = open("stemming/vocabulary2.json", "r")
    vocabulary = json.load(sf)
    sf.close()
    sf = open("stemming/DocID2.json", "r")
    docid = json.load(sf)
    sf.close()
    
    unique_token = list(vocabulary.values())

    vec = [[]]
    for item in postings:
        vec.append(myVector(postings[item], unique_token))
    norm_vectors = [normalise(vecs) for vecs in vec]

    beep=True
    while beep:
        searcher = input("ENTER QUERY: ")
        q_token = word_tokenize(searcher)
        new = []

        for item in q_token:
            # st = PorterStemmer()
            # i = st.stem(item)
            new.append(item)

        query_count = Counter(new)

        #unique_token = list(vocabulary.values())

        test = (vectorise(query_count, unique_token))

        q_tf_idf = (tf_idf_row(test, unique_token,
                    doc_freq, len(docid), vocabulary))

        norm_q_tf_idf = normalise(q_tf_idf)
        

        dots = [dot(norm_q_tf_idf, vec) for vec in norm_vectors]

        closest = max(dots)
        print("Top result: ")
        index = dots.index(closest)
        print(docid[str(index-1)])

        print("The current words that are in this document are:")

        items = postings[str(index-1)]
        for i in items:
            #test = vocabulary[str(i[0])]
            print(vocabulary[str(i[0])], end=" ")

        print("\n")
        test = heapq.nlargest(len(dots),dots)  
        print("Next results in order: ")
        
        

        ###FOR TESTING IF QUERY CONTAINS
        ntest = []
        btest = []
        count=1
        for i in test:
            closest = i
            #print(max(dots))
            index = dots.index(closest)
            if (index==0):
                break
            if count<15:
                print(count,docid[str(index-1)])
            ntest.append(docid[str(index-1)])
            btest.append(str(index-1))
            count+=1
        

        sf = open("all_words/postings_all.json", "r")
        postings2 = json.load(sf)
        sf.close()
        sf = open("all_words/vocabulary_all.json", "r")
        vocabulary2 = json.load(sf)
        sf.close()

        test = []

        for item in btest:
            fun = []
            items = postings2[item]
            val_list = list(vocabulary2.values())

            mine = items.split(",")
            
            for i in q_token:
                if i in val_list:
                    for items in mine:
                            new = items.split("|")
                            if (str(val_list.index(i)) == str(new[0])):
                                fun.append(1)
                                break
                            else:
                                fun.append(0)
            
            final = sum(fun)
            if final>=(len(q_token)/2):
                test.append(1)
            else:
                test.append(0)
        print("The relevant documents over retrieved documents: ")
        print(sum(test))
        print("/")
        print(len(test))

        zak = input("Enter if want to continue y/n: ")
        if zak=="n":
            beep=False




def main():

    check1 = input("WOULD YOU LIKE TO CREATE TABLES:1, CREATE TF IDF:2, QUERIES:3 : ")
    
    check2 = input("ARE YOU SURE y/n: ")
    if check1 == "1" and check2 == "y":
        part1()

    if check1 == "2" and check2 == "y":
        part2()

    if check1 == "3" and check2 == "y":
        part4()

main()