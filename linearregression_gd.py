import numpy as np
import matplotlib.pyplot as plt

# The first column is the index and can be discarded. The next 15 columns are␣
#input features. The last column is the output target.
data = np.loadtxt('data.txt')
X = np.array(data[:, 1:16])
y = np.array(data[:, 16], ndmin=2).T
X, y
# The number of features.
n = X.shape[1]
print("number of features:",n)
##################1.1.Noramlization of features and target variable##########################
for i in range(n):
    X[:, i] = ( X[:, i]-np.min(X[:, i]) ) / ( np.max(X[:, i])-np.min(X[:, i]) )
y = ( y-np.min(y) ) / ( np.max(y)-np.min(y) )
train_X = X[0:48, :]
test_X = X[48:60, :]
train_y = y[0:48]
test_y = y[48:60]

# The number of examples in the training set.
m = train_X.shape[0]
print("number of examples in training dataset",m)

# Add 1s into the feature matrix
train_X = np.concatenate( (np.ones((m,1)), train_X), axis=1 )
para = np.zeros((n+1,1))

lr = 0.01
ep = 0.001
la = 5
#################1.2 The gradient descent with quadratic regularization algorithm.####################
def loss(X, y, para, la):
    part = y-np.dot(X, para)
    cost = 1/(2*m)*np.dot(part.T, part) + la/(2*m)*np.dot(para.T, para)

    return cost.item()

def gradientDescent(X, y, lr, ep, la):
    m = X.shape[0]
    n = X.shape[1]
    para = np.zeros((n,1))
    new_para = np.zeros((n,1))
    costHistory = []
    costHistory.append(loss(X, y, para, la))
    k = 0
    while True:
        h = (np.dot(X, para) - y).T  # compute the predictions of all training␣
    #, →examples.
        for j in range(n):
            grad = 1 / m * np.dot(h, X[:, j]) + la / m * para[j]
            new_para[j] = para[j] - lr * grad  # note here I store the new␣
                 #, →paratemeters in a new vector.
        para = new_para
        costHistory.append(loss(X, y, para, la))
        k = k + 1
        # convergence critrion
        if abs(costHistory[k - 1] - costHistory[k]) * 100 / costHistory[k - 1] < ep:
            break
    return costHistory, para

cost, para = gradientDescent(train_X, train_y, lr, ep, la)
print("The parameters obtained by gradient descent using quadratic regularization are:",para)
# Finally print out the cost of each iteration for training dataset
plt.plot(cost)
plt.ylabel('cost')
plt.show()

############### finding squared loss function on test data w/o regularization ################
# number of examples in test dataset
p = test_X.shape[0]
print("number of examples in testdata",p)

test_X = np.concatenate( (np.ones((p,1)), test_X), axis=1 )
para2 = np.zeros((n+1,1))

# squared loss function of test dataset
def loss_test(X, y, para1):
    part1 = y-np.dot(X, para1)
    cost1 = 1/(2*p)*np.dot(part1.T, part1)
    return cost1.item()
loss_test_qr = loss_test(test_X, test_y, para2)
print("The squared loss on test data w/o quadratic reg is",loss_test_qr)

#count number of zero parameters obtained using gradient decesent with quaradtic regularization
cnt = 0
for j in range(len(para)):
  if np.absolute(para[j]) < lr:
    para[j] = 0
    cnt += 1
print("The number of zero parameters obtained using quadratic regularization is:",cnt)

#############1.3 Gradient descent with lasso regularization ##########################

# The number of examples in the training set.
m2 = train_X.shape[0]
m2
para2 = np.zeros((n+1,1))

lr2 = 0.01
ep2 = 0.001
la2 = 1

def loss2(X, y, para2, la2):
    part2 = y-np.dot(X, para2)
    cost2 = 1/(2*m2)*np.dot(part2.T, part2) + (la2/(2*m2))*sum(np.absolute(para2))
    return cost2.item()

# The gradient algorithm with lasso regularization.
def gradientDescent2(X, y, lr2, ep2, la2):
    m2 = X.shape[0]
    n2 = X.shape[1]
    para2 = np.zeros((n2,1))
    new_para2 = np.zeros((n2,1))
    costHistory2 = []
    costHistory2.append(loss2(X, y, para2, la2))
    k = 0
    while True:
        h = (np.dot(X, para2) - y).T  # compute the predictions of all training␣
    #, →examples.
        for j in range(n2):
            if para2[j] < 0:
                grad2 = 1 / m2 * np.dot(h, X[:, j]) - (la2*n2) / (2*m2)
            else:
                grad2 = 1 / m2 * np.dot(h, X[:, j]) + (la2*n2) / (2*m2)
            new_para2[j] = para2[j] - lr2 * grad2  # note here I store the new␣paratemeters in a new vector.
        para2 = new_para2
        costHistory2.append(loss2(X, y, para2, la2))
        k = k + 1
        # convergence critrion
        if abs(costHistory2[k - 1] - costHistory2[k]) * 100 / costHistory2[k - 1] < ep2:
           break

    return costHistory2, para2

cost2, para2 = gradientDescent2(train_X, train_y, lr2, ep2, la2)
print("The parameters obtained by gradient descent using lasso regularization are:",para2)
# Finally print out the cost of each iteration for training dataset
plt.plot(cost2)
plt.ylabel('cost')
plt.show()


# squared loss function of test dataset w/o lasso regularization function.
para3 = np.zeros((n+1,1))
def loss_lasso_test(X, y, para3):
    part3 = y-np.dot(X, para3)
    cost3 = 1/(2*p) * np.dot(part3.T, part3)
    return cost3.item()
loss_test_lr = loss_lasso_test(test_X, test_y, para3)
print("The squared loss on test data without lasso regularization is:",loss_test_lr)

#count no of zero parameters obtained using this model
count = 0
for j in range(len(para2)):
  if np.absolute(para2[j]) < lr2:
    para2[j] = 0
    count += 1
print("the number of zero parameters using lasso regularization:",count)





