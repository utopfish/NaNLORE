from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.cluster import KMeans
def KNN(trainX,trainY):
    clf=KNeighborsClassifier()
    clf.fit(trainX,trainY)
def Kmeans(X,target):
    y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)

    count=0
    for i in range(len(y_pred)):
        if y_pred[i]==0:
            y_pred[i]=3
    for i in range(len(y_pred)):
        if y_pred[i]==1:
            y_pred[i]=0
    for i in range(len(y_pred)):
        if y_pred[i]==3:
            y_pred[i]=1

    for i in range(len(target)):
        if target[i]==y_pred[i]:
            count+=1
    print(y_pred)
    print(count/len(target))

if __name__=="__main__":
    data=datasets.load_iris()
    Kmeans(data['data'],data['target'])