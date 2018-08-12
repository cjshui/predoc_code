import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import pickle
import matplotlib.pyplot as plt


def nbpredict(train_x,train_y,test_x,test_y):

    clf = GaussianNB()
    clf.fit(train_x,train_y)

    return 1-clf.score(test_x,test_y)


def sgdpredict(train_x,train_y,test_x,test_y):

    clf = LinearSVC(random_state=0)
    clf.fit(train_x,train_y)

    return 1-clf.score(test_x,test_y)



def nnpredict(train_x,train_y,test_x,test_y):

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(train_x,train_y)

    return 1-clf.score(test_x,test_y)




def data_test(data, sample_size = 50):

    X = data[:,:-1]
    Y = data[:,-1]



    total_iter = int(len(Y)/sample_size)

    error = np.zeros([total_iter-1,5])


    for i in range(total_iter-1):

        X_current = X[:(i+1)*sample_size-1,:]
        y_current = Y[:(i+1)*sample_size-1]

        X_next    = X[(i+1)*sample_size:(i+2)*sample_size-1,:]
        y_next    = Y[(i+1)*sample_size:(i+2)*sample_size-1]

        error[i, 0] = nbpredict(X_current,y_current,X_next,y_next)
        error[i, 1] = sgdpredict(X_current,y_current,X_next,y_next)

        current_len = len(y_current)

        if current_len <= 100:

            error[i, 2] = nnpredict(X_current, y_current, X_next, y_next)
            error[i, 3] = nnpredict(X_current, y_current, X_next, y_next)
            error[i, 4] = nnpredict(X_current, y_current, X_next, y_next)

        elif current_len > 100 and current_len <= 1500:

            error[i, 2] = nnpredict(X_current[-150:,:], y_current[-150:], X_next, y_next)
            error[i, 3] = nnpredict(X_current, y_current, X_next, y_next)
            error[i, 4] = nnpredict(X_current, y_current, X_next, y_next)

        elif current_len > 1500 and current_len <= 6000:

            error[i, 2] = nnpredict(X_current[-150:,:], y_current[-150:], X_next, y_next)
            error[i, 3] = nnpredict(X_current[-1500:,:], y_current[-1500:], X_next, y_next)
            error[i, 4] = nnpredict(X_current, y_current, X_next, y_next)

        else:

            error[i, 2] = nnpredict(X_current[-150:,:], y_current[-150:], X_next, y_next)
            error[i, 3] = nnpredict(X_current[-1500:,:], y_current[-1500:], X_next, y_next)
            error[i, 4] = nnpredict(X_current[-6000:,:], y_current[-6000:], X_next, y_next)


    return error


def test(dataname,briskname):

    # loading data
    with open(dataname, "rb") as fp:
          data = pickle.load(fp)

    # with open("syn_data/NSGT_5d.txt", "rb") as fp:
    #       data = pickle.load(fp)

    # with open("syn_data/NSGR.txt", "rb") as fp:
    #       data = pickle.load(fp)

    # with open("syn_data/NSPC.txt", "rb") as fp:
    #       data = pickle.load(fp)

    # with open("syn_data/NSPCA.txt", "rb") as fp:
    #        data = pickle.load(fp)





    # processing data
    error = data_test(data,sample_size=25)
    total_L = np.size(error,0)


    ave_error = np.zeros([20,5])
    gap = int(int(total_L/20))

    for i in range(20):

        ave_error[i,:] = np.mean(error[i*gap:(i+1)*gap-1,:],axis=0)

    with open(briskname,"rb") as fps:
        brisk = pickle.load(fps)

    # with open("result/NSGT5d_brisk.txt","rb") as fps:
    #      brisk = pickle.load(fps)

    # with open("result/NSPC_brisk.txt","rb") as fps:
    #      brisk = pickle.load(fps)

    # with open("result/NSPCA_brisk.txt","rb") as fps:
    #       brisk = pickle.load(fps)


    brisk = 0.7* np.array(brisk)
    # brisk = np.zeros(20)



    plt.figure()
    plt.plot(np.arange(500,10001,500), ave_error[:,0] ,'b',  label='NB')
    plt.plot(np.arange(500,10001,500), ave_error[:,1] ,'g',  label='SGD')
    plt.plot(np.arange(500,10001,500), ave_error[:,2], 'r',  label='NN100')
    plt.plot(np.arange(500,10001,500), ave_error[:,3], 'k',  label='NN1500')
    plt.plot(np.arange(500,10001,500), ave_error[:,4], 'm',  label='NN6000')
    plt.plot(np.arange(500,10001,500), brisk,        '--r', label='Optimal')


    plt.xlabel('Number of instances')
    plt.ylabel('Error rate')

    plt.legend()
    plt.tight_layout()
    plt.show()

    # print(np.mean(brisk))
    # print(np.mean(ave_error,axis=0))

if __name__ =="__main__":

    data_name = "syn_data/NSGT.txt"
    brisk_name = "result/NSGT_brisk.txt"

    test(dataname=data_name,briskname=brisk_name)

























