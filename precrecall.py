from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics._plot.precision_recall_curve import plot_precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
#import sklearn
# for recall


def recall(array1,total):
    count = 0
    population = []
    
    for item in array1:
        actual_positives = [1 for _ in range(total[count])]
        actual_negatives = [0 for _ in range(total[count]*100)]
        y_true = actual_positives + actual_negatives
        predict_positives = [0 for _ in range(total[count]-item[0])] +[1 for _ in range(item[0])]
        predict_negatives = [0 for _ in range(total[count]*100)]
        y_predict = predict_positives + predict_negatives 
        recall = recall_score(y_true,y_predict,average = "binary")
        #print("Recall calculated as: %.3f" % recall)
        population.append(recall)
        count+=1
    return population


def plotting(array1,array2):
    score = np.array([0.9,0.8,0.6,0.55,0.54,0.53,0.52,0.51,0.505,0.4,0.39,0.38,0.37,0.36,0.35])
    y = np.array([1,1,0,1,1,1,0,0,1,0,1,0,1,0,0,0,1,0,1,0])

    # score = np.array(array1)
    #print(score)
    #y = np.array(array2)
    print(len(y))
    print(len(score))

    fpr = []
    tpr = []

    thresholds = np.arange(0.0, 1.01, .01)
    p = sum(y)
    n = len(y)-p

    for thresh in thresholds:
        FP = 0
        TP = 0
        for i in range(len(score)):
            if (score[i]>thresh):
                if y[i]==1:
                    TP = TP +1
                if y[i]==0:
                    FP = FP +1
        fpr.append(FP/float(n))
        tpr.append(TP/float(p))
    plt.plot(fpr,tpr)
    plt.show()
    plt.scatter(fpr,tpr)
    plt.show()


def graph2(array1,array2,array3,array4):

    plt.title("Precision-Recall Graph combined")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    plt.plot(array1,array2)
    plt.plot(array3,array4)
    plt.show()

def graph(array1,array2):

    plt.title("Precision-Recall Graph for Lemmatization")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.plot(array1,array2)
    # plt.show()
    plt.plot(array1,array2)
    plt.show()



def precision(array1,total):
    count = 0
    population = []
    for item in array1:
        actual_positives = [1 for _ in range(total[count])]
        actual_negatives = [0 for _ in range(total[count]*100)]
        y_true = actual_positives + actual_negatives
        predict_positives = [0 for _ in range(total[count]-item[0])] +[1 for _ in range(item[0])]
        predict_negatives = [1 for _ in range(item[1]-item[0])]+[0 for _ in range((total[count]*100)-(item[1]-item[0]))]
        y_predicted = predict_positives + predict_negatives 
        
        precision = precision_score(y_true,y_predicted,average = "binary")
        #print("Precision calculated as: %.3f" % precision)
        population.append(precision)
        count+=1

    return population

def map(array1,array2,array3):
    MAP = []
    for item in range(len(array1)):
        x = array1[item]+array2[item]+array3[item] 
        z = x /3
        MAP.append(z)
    final = sum(MAP)
    print(final/5)




def main():
    
    stopwords = [[251,251],[1318,1318],[1859,1859],[193,193],[434,434]]
    stemming = [[220,229],[297,300],[1859,1932],[45,45],[434,521]]
    lemma = [[254,621],[1318,1320],[1859,1865],[197,199],[434,441]]
    lemma2 = [[110,240],[745,747],[1859,1868],[79,82],[434,456]]
    query = [621,2304,3602,1961,1790]
    stemavg = 605
    stopavg = 811
    lemmaavg = 889
    queryavg = 2055
    all = [450,1521,1940,210,494]

    stop_recall = recall(stopwords,all)
    stem_recall = recall(stemming,all)
    lemma_recall = recall(lemma,all)
    lemma2_recall = recall(lemma2,all)

    stop_precision = precision(stopwords,all)
    stem_precision = precision(stemming,all)
    lemma_precision = precision(lemma,all)
    lemma2_precision = precision(lemma2,all)

    #print(stop_recall,stop_precision)
    #map(stop_precision,stem_precision,lemma_precision)
    # plotting(lemma_recall,lemma_precision)
    # plotting(stem_recall,stem_precision)
    #plotting(stop_recall,stop_precision)
    graph2(lemma_recall,lemma_precision,lemma2_recall,lemma2_precision)
    #graph(stem_recall,stem_precision)
    #graph(stop_recall,stop_precision)
    # graph2(stop_recall,stopwords)
    # graph2(lemma_recall,lemma)
    # graph2(stem_recall,stemming)
    

main()  


# stopwords = [[251,251],[1318,1318],[1859,1859],[193,193],[434,434]]
    # stemming = [[220,229],[300,300],[1859,1932],[45,45],[434,521]]
    # lemma = [[254,621],[1318,1320],[1859,1865],[197,199],[434,441]]
    # all = [633,1320,1940,210,528]