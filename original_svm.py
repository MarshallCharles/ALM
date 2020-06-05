import pandas as pd
import numpy as np
import itertools
import random
# module for quadratic programming 
import cvxopt
# module for counting k-mers
from sklearn.feature_extraction.text import CountVectorizer

# upload raw data from csv
def get_raw_data(path = './data/'):
    for i in range(3):
        train = pd.read_csv(path+'Xtr'+str(i)+'.csv') if i==0 else pd.concat([train,pd.read_csv(path+'Xtr'+str(i)+'.csv')])
        test = pd.read_csv(path+'Xte'+str(i)+'.csv') if i==0 else pd.concat([test,pd.read_csv(path+'Xte'+str(i)+'.csv')])
        labels = pd.read_csv(path+'Ytr'+str(i)+'.csv') if i==0 else pd.concat([labels,pd.read_csv(path+'Ytr'+str(i)+'.csv')])
    train = pd.merge(train, labels)
    return train, test

# shuffle and split data for train and validation 75/25
def split_data(data, labels, train_size=1499, random_state=None):
    random.seed(random_state) 
    temp = list(zip(data, labels))
    random.shuffle(temp)
    data, labels = zip(*temp)
    return data[ : train_size], labels[ : train_size], data[train_size : ], labels[train_size : ]

# reformat labels (-1,1) -> (0,1)
def reformat_labels(p):
    for i in range(len(p)):
        if p[i]<0:
            p[i] = 0
        else:
            p[i] = 1
    return p
    
# compute binary classification accuracy
def compute_accuracy(predicted, truth):
    num_wrong = 0
    for p,t in zip(predicted,truth):
        if p!=t:
            num_wrong += 1
    loss = num_wrong / len(predicted)
    print("Accuracy is {0:.5f}".format(1 - loss))
    return(1 - loss)
    
# save results to csv
def save_results(ids, bounds, filename = 'Yte.csv'):
    df = pd.DataFrame()
    df['Id'] = ids 
    df['Bound'] = bounds
    with open(filename, 'a') as f:
        df.to_csv(f, mode='a', index = False, header=f.tell()==0)

# create vocabulary of K-mers 
def create_vocabulary(dictionary, kmer_size):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(kmer_size, kmer_size))
    vectorizer.fit_transform(dictionary)
    return vectorizer
    
# preprocess data w.r.t vocabulary      
def preprocess_data(data, labels = None, vectorizer=None, isTest=False):
    
    data = np.asarray(data)
    
    X = vectorizer.transform(data)
    
    if isTest:
        return X.toarray()
    else:
        Y = np.array([-1. if y==0. else 1 for y in labels]).astype(float)
        return X.toarray(),Y
    
# linear kernel function
def linear_norm_kernel(x, y):
    return np.dot(x, y.T) / np.sqrt(np.dot(x,x.T) * np.dot(y,y.T))

# rbf kernel function
def rbf_kernel(x, y, gamma = 0.01):
    return np.exp(-gamma * (np.linalg.norm(x - y) ** 2) )

#SVM classifier
class svm:

    def __init__(self, C=1., kernel=rbf_kernel, threshold=1e-11):
        
        self.C = C
        self.threshold = threshold
        self.kernel = kernel


    def compute_kernel(self,X, X_):
        
        self.n_samples = X.shape[0]

        K = np.zeros((self.n_samples, X_.shape[0]))
        
        for i in range(self.n_samples):
            for j in range(X_.shape[0]):
                K[i,j] = self.kernel(X[i], X_[j])
                
        self.K = K
        
    def fit(self,X,Y):
        
        self.compute_kernel(X, X)
        
        P = Y * Y.T * self.K
        q = - np.ones((self.n_samples, 1))
        G = np.concatenate((np.eye(self.n_samples), - np.eye(self.n_samples)))
        h = np.concatenate((self.C * np.ones((self.n_samples,1)),np.zeros((self.n_samples, 1))))
        A = Y.reshape(1, self.n_samples)
        b = 0.0

        # Solve QP problem
        solution = cvxopt.solvers.qp(cvxopt.matrix(P),cvxopt.matrix(q),cvxopt.matrix(G),
                                cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))
        # Lagrange multipliers
        lambdas = np.array(solution['x'])

        self.sup_vec = np.where(lambdas > self.threshold)[0]
        self.num_vec = len(self.sup_vec)
        self.X = X[self.sup_vec,:]
        self.lambdas = lambdas[self.sup_vec]
        self.Y = Y[self.sup_vec]
        self.b = np.sum(self.Y)
        for n in range(self.num_vec):
            self.b -= np.sum(self.lambdas * self.Y * np.reshape(self.K[self.sup_vec[n], self.sup_vec],(self.num_vec, 1)))
        self.b /= len(self.lambdas)
            
    def predict(self,X):
        
        self.compute_kernel(X, self.X)       
   
        self.y = np.zeros((self.n_samples,1))
        for j in range(self.n_samples):
            for i in range(self.num_vec):
                self.y[j] += self.lambdas[i] * self.Y[i] * self.K[j,i]
            self.y[j] += self.b

        return np.sign(self.y).reshape((1,-1))[0]

       
def tuning_parameters(K=None,
                      kernel=None,
                      C=None,
                      threshold=None,
                      offset=None, # for different datasets 0:0, 1:2000, 2:4000
                      random_states=None):
    
    train_size = 1599 # size of the validation set is 1/5 of the training set (5-folds)
    
    size = 2000 # size of each dataset  
    
    train, test = get_raw_data()

    vectorizer = create_vocabulary(train['seq'][offset : offset + size], K)
    
    average_accuracy = 0
    
    for random_state in random_states:
    
        tr,tr_lab, val,val_lab = split_data(train['seq'][offset : offset + size],
                                            train['Bound'][offset : offset + size],
                                            train_size=train_size,
                                            random_state=random_state)

        X_train, Y_train =  preprocess_data(tr, labels=tr_lab, vectorizer=vectorizer)        
       
        print("Training SVM...")
        clf = svm(C=C, kernel=kernel, threshold=threshold)
        clf.fit(X_train,Y_train.reshape((-1,1)))
        print("Training is finished!")

        print("Predicting validation labels...")
        X_val =  preprocess_data(val, vectorizer=vectorizer, isTest=True) 
        prediction = clf.predict(X_val)
        prediction = reformat_labels(prediction)
        accuracy = compute_accuracy(prediction, val_lab)
        
        average_accuracy += accuracy

    print('Average accuracy is : {0:.5f}'.format(average_accuracy / len(random_state)))

    
def main():
#     accuracies = []
#     I = [0, 1, 2] 
#     offsets = [0, 2000, 4000]
#     K = [3, 4, 5, 6, 7, 8]
    
#     C = 100
#     gamma = 0.01 # already in rbf_kernel function
#     threshold = 1e-3
    
#     random_states = [0,42,17,69,324] # 5 random states for 5-folds cross-validation
    
    
#     for i, offset, k in zip(I, offsets, K):
#         print("####################DATASET: {}####################".format(i))
#         print('k-mer: {}'.format(k))
#         accuracies.append((k, tuning_parameters(K=k, kernel=rbf_kernel,  C=C, threshold=threshold, offset=offset, random_states=random_states)))
  
#     f = open("results.txt","a") 
#     f.write(str([i, accuracies])) 
#     f.close() 

    # get the data
    train, test = get_raw_data()
    
    # set all parameters
    I = [0, 1, 2]  # dataset indices
    offsets = [0, 2000, 4000] # corresponding offsets for corresponding datasets 0:0, 1:2000, 2:4000
    K = [6, 6, 6] # sizes of the k-mers
    C = [10, 10, 10] # C parameter, (gamma is already set in the function)
    threshold = 1e-3 # threshold for support vectors
    # we use rbf_kernel!
    
    # run the model
    for i, offset, k, c in zip(I, offsets, K, C):
        offset_train = offset
        offset_test = int(offset / 2)
        size_train = 2000
        size_test = 1000
         
        vectorizer = create_vocabulary(train['seq'][offset_train : offset_train + size_train].append(test['seq'][offset_test : offset_test + size_test]), k)
        
        print("####################DATASET:{}####################".format(i))
        print("Preprocessing training data...")
        X_train, Y_train =  preprocess_data(train['seq'][offset_train : offset_train + size_train], 
                                            labels=train['Bound'][offset_train : offset_train + size_train], 
                                            vectorizer=vectorizer)
        print("Preprocessing is finished!")
        print("Training SVM...")
        clf = svm(C=c, kernel=rbf_kernel, threshold=threshold)
        clf.fit(X_train,Y_train.reshape((-1,1)))
        print("Training is finished!")
        
        print("Preprocessing test data...")
        X_test = preprocess_data(test['seq'][offset_test : offset_test + size_test], vectorizer=vectorizer, isTest=True)
        print("Preprocessing is finished!")
        print("Predicting test labels...")
        prediction = clf.predict(X_test)
        prediction = reformat_labels(prediction)
        save_results(test['Id'][offset_test : offset_test + size_test], prediction.astype(int))
        print("Results are saved in Yte.csv!")

if __name__=='__main__':
    main()
