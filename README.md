# search_engines


Local information retrieval systems that indexes words largely from html scraped from internet and uses cosine similarity for a person to search for files



Must replace files directory with local files of own.

## Files:

Each file is named after different techniques and experiments including:  
- Lemmatization using NLTK and SpaCy
- Removing Stop Words which are commonly used words such as "the" and "and".
- Using Stemming to shorten words (more aggressive than lemmatization)
- Named entity recognition
- Added weighting to special words to increase precision of searches
- Includes calculation file for the calculations of precision and recall.


## Instructions: 


User must first change directory to directory of files wanted for use to build the index! (This can take a long time based on the size of files)


This builds an inverted index of tables that includes vocabulary and file locations, then it can run tf idf calculations and finally run the search function which will return a ranked list of documents that are ranked in relevancy to query


## Screenshots:

This was to conduct analysis of different types of ways to efficiently return a ranked list of documents based on a search query, including findings here:




<img src="https://github.com/zakb27/search_engines/blob/main/images/Picture%201.jpg">

=================================================================================================================================================================================================================================================================================================================================================================================================================================================================================

<img src="https://github.com/zakb27/search_engines/blob/main/images/Picture%203.png">


