import numpy as np
from sklearn import datasets
from sklearn.neighbors import KDTree
from dijkstra import dijkstra
import math
import sys
from cluster import *
from plot import *
class NaNLOREAl():
    supk=0
    data=0
    target=0
    def getData(self):
        '''
        	获取数据集

        	Args:
        		暂无
        	返回：
        	    暂无
        	'''
        import pandas as pd
        # 读取数据
        path = r"I:\人才才能预测论文\Human_performance.xlsx"
        data_0 = pd.read_excel(path, sheet_name='dataset')
        data = np.array(data_0)  # 转换成numpy类型，float类型

        print('Loading data...')
        #
        # self.data = data[:, :-1]  # x表示数据特征
        # self.target = data[:, -1]  # y表示标签
        iris=datasets.load_iris()
        # cancer=datasets.load_breast_cancer()
        # loadData=datasets.load_iris()
        # # self.data=iris['data']
        # # self.traget=iris['target']
        self.data=iris['data']
        self.target=iris['target']
    def NaNSearching(self):
        '''
        NaNSearching算法
        参数：
            暂无
        返回：
            supK:
            NN:带每个点的最supk个临近节点
            nb:

        '''

        kdt=KDTree(self.data,leaf_size=int(len(self.data)*1.5),metric="euclidean")
        r=1
        nb=np.zeros((len(self.data),1))
        NN=[[] for i in range(len(self.data))]
        RNN=[[] for i in range(len(self.data))]
        oldNubm=0
        while True:
            #k值不同时最邻近的值的顺序不同，待解
            for i in kdt.query(self.data,k=r+1,return_distance=False):
                nb[i[r]]+=1
                NN[i[0]].append(i[r])
                RNN[i[r]].append(i[0])
            newNumb=0
            for i in nb:
                if i==0:
                    newNumb+=1
            if newNumb==oldNubm:
                break
            r+=1
            oldNubm=newNumb
        self.supk=r
        for i in range(len(self.data)):
            NN[i].append(i)
        return r,nb,NN
    def getDensity(self,NN):
        '''
        求每个点的密度
        :param NN 每个点的临近点集合:
        :return:每个点的密度
        '''
        p=np.arange(len(self.data)+1)
        p[0]=-1
        for i in range(len(self.data)):
            for j in NN[i]:
                p[i+1]+=np.sqrt(np.sum(np.square(self.data[i]-self.data[j])))
        return self.supk/p
    def LORE(self,NN,rho):
        '''
        通过密度找到local represnets
        :param NN 点的临近集合:
        :param rho 每个点的密度:
        :return:
        Re： 表面每个点所属的local represent的集合
        localRE：local represent 的集合
        cl：根据localRe对所有点进行的简单划分
        '''
        Re=[[] for i in range(len(self.data))]
        localRe=np.zeros(len(self.data))-1

        cl=[[] for i in range(len(self.data))]

        for i in range(len(self.data)):
            maxDensityIndex=self.max(rho,NN[i])
            for j in NN[i]:
                if type(Re[j])==list:
                    Re[j]=(maxDensityIndex)
                if type(Re[j])!=list and Re[j]!=maxDensityIndex:
                    if rho[Re[j]]<rho[maxDensityIndex]:
                        Re[j]=maxDensityIndex
                for z in range(len(self.data)):
                    if Re[z]==int(j):
                        Re[z]=Re[int(j)]
        for i in range(len(self.data)):
            maxDensityIndex=self.max(rho,NN[i])
            for j in NN[i]:
                if type(Re[j])==list:
                    Re[j]=(maxDensityIndex)
                if type(Re[j])!=list and Re[j]!=maxDensityIndex:
                    if rho[Re[j]]<rho[maxDensityIndex]:
                        Re[j]=maxDensityIndex
                for z in range(len(self.data)):
                    if Re[z]==int(j):
                        Re[z]=Re[int(j)]
        K=0
        for x in range(len(self.data)):
            if(Re[x]==x):
                localRe[K]=x
                cl[x]=K
                K+=1

        for x in range(len(self.data)):
            cl[x]=cl[int(Re[x])]
        localRe=self.Cut(localRe)
        for i in range(len(localRe)):
            localRe[i]=localRe[i]
        return Re,localRe,cl

    def NaNLORE(self,localRe,Re):
        '''
        整理dp算法需要的参数
        :param localRe:
        :param Re:
        :return:
        '''
        Matrix=np.zeros((len(localRe),len(localRe)))
        mulriple=self.densitySensitiveTest(localRe)
        if(mulriple==0):
            for i in range(len(localRe)):
                for j in range(i, len(localRe)):
                    Matrix[i, j] = self.densitySensitiveLineLength(int(localRe[i]), int(localRe[j]))
                    Matrix[j, i] = Matrix[i, j]
        else:
            for i in range(len(localRe)):
                for j in range(i,len(localRe)):
                    Matrix[i,j]=self.densitySensitiveLineLength2(int(localRe[i]),int(localRe[j]),mulriple)
                    Matrix[j,i]=Matrix[i,j]



        reDistance={}
        reRho=np.ones(len(localRe)+1,dtype=float)
        reRho[0]=-1
        for index,value in enumerate(localRe):
            reRho[index+1]=float(rho[int(value)])

        min_dis, max_dis = sys.float_info.max, 0.0
        max_id = 0
        for i in range(len(Matrix.tolist())):
            distance,path=dijkstra(Matrix.tolist(),i)
            for j in range(len(Matrix.tolist())):
                reDistance[(i+1,j+1)]=distance[j]
                reDistance[(j+1,i+1)]=distance[j]
                max_id = max(max_id, i+1, j+1)
                min_dis, max_dis = min(min_dis, distance[j]), max(max_dis, distance[j])

        delta, nneigh = min_distance(max_id, max_dis, reDistance, reRho)

        #plot_rho_delta(reRho, delta)
        # plot to choose the threthold

        return reDistance,reRho,max_id,max_dis,min_dis,reRho,delta
    def dpAl(self,distances,rho, max_id, max_dis, min_dis,density_threshold, distance_threshold, dc=None, auto_select_dc=False):
        Recl = {}
        dpcluster = DensityPeakCluster()
        dpcluster.cluster2(distances,rho, max_id, max_dis, min_dis,density_threshold, distance_threshold, dc=None, auto_select_dc=False)
        print("聚类中心如下")
        print(dpcluster.ccenter)
        print("聚类结果如下")
        print(dpcluster.cluster)
        try:
            plot_cluster(dpcluster)
        except:
            print("聚类绘图报错")
        for key,value in dpcluster.cluster.items():
            if(value!=-1):

                Recl[int(localRe[key-1])]=int(localRe[value-1])
            else:
                Recl[int(localRe[key]-1)] =-1

        for index in range(len(cl)):
            cl[index] = Recl[Re[index]]

        mark = {}
        for index, value in enumerate(set(cl)):
            if (value != -1):
                mark[value] = index
        rep = [mark[x] if x in mark else x for x in cl]
        return rep
    def score(self,rep):
        #现在还有bug，对分类结果需要进行一些手动调整
        tar = list(self.target)
        count = 0
        print("算法聚类结果如下：")
        print(rep)

        #下面用来交换标签1，0的位置
        for i in range(len(rep)):
            if rep[i]==1:
                rep[i]=3
        for i in range(len(rep)):
            if rep[i]==0:
                rep[i]=1
        for i in range(len(rep)):
            if rep[i]==3:
                rep[i]=0
        print('标签值如下：')
        print(self.target)
        for i in range(len(cl)):
            if rep[i] == tar[i]:
                count += 1
        print("分类准确率")
        print(count / len(cl))

        if (len(set(tar)))==2:
            TP, TN, FP, FN=0,0,0,0
            for i in range(len(rep)):
                if rep[i]==tar[i] and rep[i]==0:
                    TP+=1
                elif rep[i]==tar[i] and rep[i]==1:
                    TN+=1
                elif rep[i]==1 and tar[i]==0:
                    FP+=1
                elif rep[i]==0 and tar[i]==1:
                    FN+=1
            print("F1值：")
            print(self.f1(TP, TN, FP, FN))
    ###tools方法
    def densitySensitiveTest(self,localRe):
        '''
        解决欧式距离过大，导致在计算距离适应矩阵时数据溢出的问题
        :param localRe:
        :return:
        '''
        eucli=np.zeros((len(localRe),len(localRe)))
        for i in range(len(localRe)):
            for j in range(len(localRe)):
                eucli[i,j]=np.sqrt(np.sum(np.square(self.data[int(localRe[i])] - self.data[int(localRe[j])])))
        maxVaule=(max(max(eucli.tolist())))
        if maxVaule>300:
            return maxVaule/300
        else:
            return 0
    def cluster(self,density_threshold,distance_threshold,delta,nneigh,reRho,max_id):
        ccenter,cluster={},{}
        for idx, (ldensity, mdistance, nneigh_item) in enumerate(zip(reRho, delta, nneigh)):
            # if idx == 0:
            #     continue
            if ldensity >= density_threshold and mdistance >= distance_threshold:
                ccenter[idx] = idx
                cluster[idx] = idx
        for idx, (ldensity, mdistance, nneigh_item) in enumerate(zip(reRho, delta, nneigh)):
            if idx == 0 or idx in ccenter:
                continue
            if nneigh_item in cluster:
                cluster[idx] = cluster[nneigh_item]
            else:
                cluster[idx] = -1
            if idx % (max_id / 10) == 0:
                logger.info("PROGRESS: at index #%i" % (idx))
        return cluster
        # self.cluster, self.ccenter = cluster, ccenter
    def densitySensitiveLineLength(self,pointX,pointY):
        #p论文中默认参数为2
        p=2
        euclideanDistance=np.sqrt(np.sum(np.square(self.data[pointX] - self.data[pointY])))
        L=math.pow(math.exp(p*euclideanDistance-1),1/p)
        return L
    def densitySensitiveLineLength2(self,pointX,pointY,mulriple):
        #p论文中默认参数为2
        p=2
        euclideanDistance=np.sqrt(np.sum(np.square(self.data[pointX] - self.data[pointY])))
        euclideanDistance=euclideanDistance/mulriple
        L=math.pow(math.exp(p*euclideanDistance-1),1/p)
        return L
    def Cut(self,cl):
        #去掉数组后多余的数字
        index=0
        for i in range(len(cl)):
            if cl[i]==-1:
                index=i
                break
        return cl[:index]
    def max(self,rho,NNi):
        #找到邻居中rho值最大的点
        maxValue=0
        maxIndex=0
        for i in NNi:
            if(rho[i]>maxValue):
                maxIndex=i
                maxValue=rho[i]
        return maxIndex
    def f1(self,TP,TN,FP,FN):
        '''
        二分类时使用
        :param TP:
        :param TN:
        :param FP:
        :param FN:
        :return:
        '''
        precision=(TP)/(TP+FP)
        recall=(TP)/(TP+FN)
        f1=2*precision*recall/(precision+recall)
        return f1

if __name__=="__main__":
    test=NaNLOREAl()
    #根据getData设计数据与数据标签
    test.getData()
    r, nb, NN=test.NaNSearching()
    rho=test.getDensity(NN)

    Re, localRe, cl=test.LORE(NN,rho)

    distances, rho, max_id, max_dis, min_dis,rho,delta=test.NaNLORE(localRe,Re)
    print("rho值")
    print(rho)
    print("delta值")
    print(delta)
    plot_rho_delta(rho, delta)
    #打印rho-delta图，用来设定dp算法中的参数
    result=test.dpAl(distances,rho, max_id, max_dis, min_dis,0,3)

    
    #score函数还有bug，聚类的标签不能自动调整，需要再score函数中手动调整
    #f1函数当为二分类时，算出TP，TN，FP，FN后，代入f1函数即可
    test.score(result)

