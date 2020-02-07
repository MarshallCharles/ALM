import pandas as pd
import numpy as np
import altair as alt
from termcolor import colored

MAX_ETA = 0.25
NITER = 1000
DATA_MAP = {'G':1 , 'A':2 ,'C':3,'T':4}
def get_data(path = './data/'):
    for i in range(3):
        train = pd.read_csv(path+'Xtr'+str(i)+'.csv') if i==0 else pd.concat([train,pd.read_csv(path+'Xtr'+str(i)+'.csv')])
        test = pd.read_csv(path+'Xte'+str(i)+'.csv') if i==0 else pd.concat([test,pd.read_csv(path+'Xte'+str(i)+'.csv')])
        labs = pd.read_csv(path+'Ytr'+str(i)+'.csv') if i==0 else pd.concat([labs,pd.read_csv(path+'Ytr'+str(i)+'.csv')])
    train = pd.merge(train, labs)
    return train, test

def seq_distrib(series):
    dic = {}
    for item in series:
        for sub in item:
            if sub in dic:
                dic[sub] += 1
                continue
            dic[sub] = 1
    return dic

class SVM:
    def __init__(self, learning_rate = 0.01, regularizer = 0.01, iterations = 1000):
        self.eta = learning_rate if isinstance(learning_rate, list) else [learning_rate] * iterations
        self.regularizer = regularizer
        self.niter = iterations

    @staticmethod
    def to_numeric(seq):
        return [DATA_MAP[x] for x in seq]

    def linear_model(self, x):
        return np.dot(self.W, x) - self.b

    def hinge(self, x,y):
        if y * self.linear_model(x) >= 1:
            return 1
        return 0

    def total(self):
        loss = 0
        for n in range(len(self.X)):
            loss += hinge(self.X[i], self.Y[i])
        return loss / len(self.Y)

    def fit(self, data, labels):
        '''Initialize samples X, labels Y, weights W and bias b'''
        self.X = np.array([self.to_numeric(x) for x in data])
        n_samples, n_features = self.X.shape
        self.Y = np.where(np.array(labels) <=0,-1,1)
        self.W = np.zeros(n_features)
        self.b = 0

        for i in range(self.niter):
            # if i%100==0:
            #     print("Iteration {}".format(i))
            for idi,xi in enumerate(self.X):
                if self.hinge(xi, self.Y[idi]):
                    self.W -= self.eta[i] * (2 * self.regularizer * self.W)
                else:
                    self.W -= self.eta[i] * (2 * self.regularizer * self.W - np.dot(xi, self.Y[idi]))
                    self.b -= self.eta[i] * self.Y[idi]

    @staticmethod
    def reformat(p):
        for i in range(len(p)):
            if p[i]<0:
                p[i] = 0
            else:
                p[i] = 1
        return p

    def predict(self,samples):
        predictions = []
        samples = np.array([self.to_numeric(x) for x in samples])
        for xi in samples:
            predictions.append(np.sign(self.linear_model(xi)))
        return self.reformat(predictions)

def split_data(data, labels):
    return data[:4999], labels[:4999], data[4999:], labels[4999:]

def batches(svm, tr, lab, n_batches = 5):
    for i in range(n_batches):
        print(colored('Batch {}'.format(i+1),'magenta'))
        svm.fit(tr, lab)
    return svm

def save_results(ids, bounds, name = 'Yte.csv'):
    print(colored('Saving results in {}'.format(name),'green'))
    df = pd.DataFrame()
    df['Id'] = ids
    df['Bound'] = bounds
    df.to_csv(name, index = False)

def accuracy(predicted, truth):
    num_wrong = 0
    for p,t in zip(predicted,truth):
        if p!=t:
            num_wrong += 1
    print(colored("pct wrong is {}".format(accuracy := (num_wrong / len(predicted))),'cyan'))
    return accuracy

def main():
    '''Initialize eta to decay, and all options for regularization parameters '''
    etas = list(np.arange(MAX_ETA,0,-1*(MAX_ETA / NITER)))
    regs = list(np.arange(1,0,-1*(1 / 15)))
    '''Get all data, split training into train/validate '''
    tr, te = get_data()
    # tr,trlab,val,vallab = split_data(tr['seq'],tr['Bound'])
    #
    # '''Run linear SVM 15 times, 5 batches each to find the best regularization parameter '''
    # cur_wrong = 1
    # which = None
    # for i in range(15):
    #     svm = SVM(learning_rate = etas, regularizer = regs[i])
    #     svm = batches(svm,tr,trlab)
    #     pct_wrong = accuracy(svm.predict(val),vallab)
    #     if pct_wrong <= cur_wrong:
    #         cur_wrong = pct_wrong
    #         which = i
    #         print(colored('Updating the best perorming svm','yellow'))
    #
    # print("Best performer is index {}, reg = {}".format(which, regs[which]))
    # print(colored('Training new SVM using all data with reg = {}'.format(regs[which]),'green'))

    '''Finally, train an SVM on all training data using the best performing regularization parameter'''
    final_svm = SVM(learning_rate = etas,regularizer = regs[-1])
    final_svm = batches(final_svm, tr['seq'],tr['Bound'])
    predictions = final_svm.predict(te['seq'])
    save_results(te['Id'],predictions)


if __name__=='__main__':
    main()
