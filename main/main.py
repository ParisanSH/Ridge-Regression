import pandas as pd 
import numpy as np
import random
import matplotlib.pyplot as plt 
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error

def RidgeRegression(x,y):
    w_lambda=[]
    df_lambda=[]
    #The penalization coefficient (λ) is varied 
    #and the feature weights are plotted against degree of freedoms 
    #for values of λ ranging from 0 to 5000.
    for i in range(0,5001,1):
        lam_par = i
        xtranspose = np.transpose(x)
        xtransx = np.dot(xtranspose , x)
        lamidentity = np.identity(xtransx.shape[0]) * lam_par
        matinv = np.linalg.inv(lamidentity + xtransx)
        xtransy = np.dot(xtranspose, y)
        w_ = np.dot(matinv, xtransy)
        #----------------------------
        _, S, _ = np.linalg.svd(x) 
        #print (S)
        #df(λ)=trace [X inv(XTransX+λI) XTranspose]
        df_ = np.sum(np.square(S) / (np.square(S) + lam_par)) 
        w_lambda.append(w_)
        df_lambda.append(df_)
    return(w_lambda,df_lambda)

def RidgeRegression_2(x_t , y_t , x_v , y_v):
    
    lam_par = 0.125
    lambda_history = []
    lambda_history.append(lam_par)
    x_t = np.asmatrix(x_t)
    y_t = np.asmatrix(y_t)
    x_v = np.asmatrix(x_v)
    y_t = np.asmatrix(y_t)
    xtranspose = np.transpose(x_t)
    xtransx = np.dot(xtranspose , x_t)
    #----------------------------
    x_v_trans = np.transpose(x_v)
    y_v_trans = np.transpose(y_v)
    y_t_trans = np.transpose(y_t)
    #----------------------------
    lamidentity = np.identity(xtransx.shape[0]) * lam_par
    matinv = np.linalg.inv(lamidentity + xtransx)
    xtransy = np.dot(xtranspose, y_t_trans)
    W_lam = np.dot(matinv, xtransy)
    #w_lam_history = []
    #w_lam_history.append(W_lam)
    #----------------------------
    x_v_trans = np.transpose(x_v)
    y_v_trans = np.transpose(y_v)
    y_t_trans = np.transpose(y_t)
    #----------------------------
    U , S, V= np.linalg.svd(x_t) 
    #----------------------------
    V_t = np.transpose(V)
    D = np.diag(1/(S+lam_par))
   # M_inv = U @ D @ V
    A = 2*(V @ xtranspose)
    A =  A @ y_t_trans
    B = (V @ x_v_trans @ x_v @ V_t)
    q = y_v_trans @ x_v @ V_t
    r = y_t @ x_t @ V_t
    #----------------------------
    d_L = (q - (r @ D @ B)) @ D @ D @ A
    dL_history = []
    dL_history.append(d_L)
    flag = 1
    for iterate in range(5000):
        if flag == -1:
            lam_par = lam_par - (random.uniform(0.005,0.35))
        else:
            lam_par = lam_par + 1
        lambda_history.append(lam_par)
        lamidentity = np.identity(xtransx.shape[0]) * lam_par
        matinv = np.linalg.inv(lamidentity + xtransx)
        xtransy = np.dot(xtranspose, y_t_trans)
        W_lam = np.dot(matinv, xtransy)
        #w_lam_history.append(W_lam)
        # compute dL/dlambda -----
        # now try to compute -> dl = (q - rDB)DDA
        D = np.diag(1/(S+lam_par))
        #M_inv = U @ D @ V
        A = 2*(V @ xtranspose @ y_t_trans )
        B = (V @ x_v_trans @ x_v @ V_t)
        q = y_v_trans @ x_v @ V_t
        r = y_t @ x_t @ V_t
        d_L_new = (q - (r @ D @ B)) @ D @ D @ A
        dL_history.append(d_L_new)
        if d_L_new > d_L:
            flag =-1
        else:
            flag = 1
        d_L = d_L_new
        #print(d_L)

    return(W_lam,dL_history,lambda_history ,lam_par)

#Degree of Freedom Plots with varying hyperparameter,  λ
def makeDFPlots(dfArray, w_lambda):
    #print wRR_array.shape, df_array.shape
    plt.figure()
    colors = ['red','blue','green','purple','orange','yellow']
    labels = ["X1=the transaction date ", "X2=the house age ", "X3=the distance to the nearest MRT", "X4=the number of convenience stores  ", "X5", "X6"]
    for i in range(0, w_lambda[0].shape[0]):
        plt.plot(dfArray, w_lambda[:,i], c = colors[i])
        plt.scatter(dfArray, w_lambda[:,i], c = colors[i], s = 8, label=labels[i])
    # df(lambda)
    plt.xlabel(r"df($\lambda$)")
    plt.legend(loc='lower left')
    plt.show()

def plotRMSEValue(max_lamda, RMSE_list):
    plt.plot(range(len(RMSE_list)), RMSE_list)
    plt.scatter(range(len(RMSE_list)), RMSE_list, s = 8)
    # df(lambda)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("RMSE")
    plt.title(r"RMSE vs $\lambda$ values for the test set, $\lambda$ = 0..%d"%(max_lamda))

def closed_form(X , Y):
    x_transpose = np.transpose(X)
    xtransx = np.dot(x_transpose,X)
    #lambda_i = np.identity(xtransx.shape[0])
    inv_xtransx = np.linalg.inv(xtransx)
    xtransy =np.dot(x_transpose , Y)
    w = np.dot(inv_xtransx , xtransy)
    return(w)

def getRMSEValues(X_test, y_test, w_lambda, max_lamda):
    RMSE_list = []
    for lamda in range(0, max_lamda+1):
        wRRvals = w_lambda[lamda]
        y_pred = np.dot(X_test, wRRvals)
        #RMSE = np.sqrt(np.sum(np.square(y_test - y_pred))/len(y_test))
        RMSE = np.sum(np.square(y_test - y_pred))/len(y_test)
        RMSE_list.append(RMSE)
    plotRMSEValue(max_lamda, RMSE_list)

#-----main----
Real_State = pd.read_csv('C://Users//Parisan.Sh//Desktop//pattern//Real_State.csv' , encoding='ansi' )
#print (Real_State)

Real_State.drop('No', axis = 1 , inplace = True)
#print(Real_State)
y = Real_State.Y
y -= y.mean()

Real_State_Normalize = scale(Real_State)
Real_State = pd.DataFrame(Real_State_Normalize , index = Real_State.index,columns = Real_State.columns )
#print(Real_State)
x = Real_State.drop('Y', axis=1)

x_train_valid , x_test , y_train_valid , y_test = train_test_split(x , y , test_size = .2 , random_state =42)
x_train , x_valid , y_train , y_valid = train_test_split(x_train_valid , y_train_valid , test_size = 0.125 , random_state =42)

#linearReg ----------------------------------------------
w_train_valid = closed_form(x_train_valid , y_train_valid)
w_train = closed_form(x_train , y_train)
w_valid = closed_form(x_valid , y_valid)

ytilda_train_valid = np.dot(x_train_valid , w_train_valid)
ytilda_train = np.dot(x_train , w_train)
ytilda_valid = np.dot(x_valid , w_valid)

mse_train_valid = mean_squared_error(y_train_valid , ytilda_train_valid)
mse_train = mean_squared_error(y_train , ytilda_train)
mse_valid = mean_squared_error(y_valid , ytilda_valid)

y_predict_for_test = np.dot(x_test , w_train_valid)
mse_test = mean_squared_error(y_test , y_predict_for_test)

print('\n',w_train_valid,'\n', w_train,'\n', w_valid)
print(mse_train_valid,'\t',mse_train,'\t',mse_valid,'\t',mse_test)
#--------------------------------------------------------
w_lambda_train, df_lambda_train = RidgeRegression(x_train,y_train)
w_lambda_train_array = np.asarray(w_lambda_train)
df_lambda_train_array = np.asarray(df_lambda_train)

w_lambda_valid, df_lambda_valid = RidgeRegression(x_valid,y_valid)
w_lambda_valid_array = np.asarray(w_lambda_valid)
df_lambda_valid_array = np.asarray(df_lambda_valid)

makeDFPlots(df_lambda_train_array, w_lambda_train_array)
makeDFPlots(df_lambda_valid_array, w_lambda_valid_array)

plt.figure()
getRMSEValues(x_test, y_test, w_lambda_train_array, max_lamda=50)
plt.show()
w_lam,dL_history, lam_history ,lam_par =RidgeRegression_2(x_train,y_train,x_valid , y_valid)
y_predict_for_test = np.dot(x_test , w_lam)
mse_test = mean_squared_error(y_test , y_predict_for_test)
plt.xlabel(r"$\lambda$")
plt.ylabel('MSE axis')
plt.grid()
