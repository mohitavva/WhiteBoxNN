import pandas as pd
import numpy as np

class nn:
    def __init__(self,ip,op):
        self.ip=ip
        self.op=op
        print(ip.shape,op.shape)
        self.l1_size=200
        self.w1=np.random.rand(self.l1_size,self.ip.shape[1]+1).T*0.3-0.15
        self.w2=np.random.rand(self.op.shape[1],self.l1_size+1).T*0.3-0.15
        print(self.w1.shape,self.w2.shape)
        for i in range(50):
            print(i)
            self.train()
    
    def sig(self,x):
        return 1/(1+np.exp(-x))
    
    def train(self):
        #===front propogation===#
        ip=np.append(np.ones((len(self.ip),1)),self.ip,axis=1)
        l1=self.sig(ip@self.w1)
        l1=np.append(np.ones((len(l1),1)),l1,axis=1)
        pred=self.sig(l1@self.w2)
        #===back propogation===#
        #error at each layer
        d3 = (pred - self.op)
        d2 = (d3 @ self.w2.T * l1 * (1-l1))
        d2 = d2[:,1:]
        #altering weights
        self.w2-= (l1.T @ d3)/len(ip)
        self.w1-= (ip.T @ d2)/len(ip)
        
    def predict(self,ip):
        ip=np.append(np.ones((len(ip),1)),ip,axis=1)
        l1=self.sig(ip@self.w1)
        l1=np.append(np.ones((len(l1),1)),l1,axis=1)
        pred=self.sig(l1@self.w2)
        return pred

def accuracy(ytst,ypred):
    print("mean error:",np.mean(np.abs((ypred-ytst).astype(bool))))
    print("rmse:",np.sqrt(((ypred-ytst).astype(bool)**2).mean()))
    
data = pd.read_csv('flight_data.csv', sep = ",")
print(data)
data.dropna()

x = data['DepDelayMinutes']
x = x.transpose()
x = x / 100
y = data['ArrDelayMinutes']



for i in range(len(x)):
    y[i,int(y[i])]=1
print(x.shape)
xtr = x.iloc[:1000,:]
ytr = y[:1000]
xtst = x.iloc[1000:,:]
ytst = y[1000:]
ytst_data = y[1000:]
net=nn(xtr,ytr)
pred=net.predict(xtst)
pred_data = np.argmax(pred,axis=1)
print("\n numpy neural network with one hidden layer")
accuracy(ytst_data,pred_data)


