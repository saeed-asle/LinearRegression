import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math



lst=[]
for i in range(9):
    lst.append("x"+str(i)) #lst of cols name to add it for thhe data cols
lst.append("y")
data=pd.read_csv('C:\\Users\\Saeed\\Desktop\\deap learing and mchine learning\\all_about_machine_and_deep_learning\\ex1\\cancer_data.csv',names=lst)

data=data.dropna()#drop null or  Dirty Data form the data
data =(data-data.mean())/(data.std())#if thhe value =mean so we get 0 ,if  data-data.mean()=std so we get 1
print(data.describe())#geting all information about data and as shown the mean is 0 and std is 1

data.insert(0,'ones',1) #add  col of ones to data

cols=data.shape[1]
X=data.iloc[:,0:cols-1]#split the data to x any y ,y is the last col in data
y=data.iloc[:,cols-1:cols]
X=np.matrix(X.values)#conver x,y to matrixs
y=np.matrix(y.values)
theta=np.matrix(np.zeros(X.shape[1]))# theta vector with size of x cols 

def h_theta(x,theta):
    return x.dot(theta.T)#calculate h(theta)=x*theta transpose

def computeCost(X,y,theta):
    z=np.power((h_theta(X,theta)-y),2) #(1/2M)*sum(h(theta)-y)^2
    return np.sum(z)/(2*len(y))

def total_j(X,y,theta):
    h=h_theta(X,theta)
    temp=X.T.dot(h-y)/ len(y)#x.transpose*(h(theta)-y)*(1/m)
    return temp.T

## every algorthim has his own code just delete(""") and run##
"""
def gradientDescent(X, y, theta, alpha, iters):
    cost = np.zeros(iters)
    for i in range(iters):
        temp =total_j(X,y,theta)# CALCULATE TOTALL J
        theta=theta-alpha*temp#update thata
        cost[i] = computeCost(X, y, theta)#save the values
    return theta, cost

alpha = 0.1
iters = 1000
start_time = time.time()#start to calculate the time that spent the func whhe we call it
theta_res,cost = gradientDescent(X, y, theta, alpha, iters)
end_time = time.time()# when the func ends
total_time= end_time - start_time#get the result of the time
time_list=np.matrix([i*total_time/iters for i in range(iters)])#divide thhe time to steps to plot the chart 
fig, ax = plt.subplots()
ax.plot(time_list.T,cost,'r')
ax.set_xlabel('time')
ax.set_ylabel('Cost')
ax.set_title('Cost vs. time')
print("time taken:",total_time)
print(theta_res)#the result of theta vector 
"""

"""
def mini_batch(X, y, theta, alpha, iters,batch_size):
    temp = np.matrix(np.zeros(theta.shape))
    cost = np.zeros(iters)
    for i in range(iters):
        for j in range(0, X.shape[0], batch_size):#like gradientDescent but with batchs
            temp =total_j(X[j:j+batch_size,:],y[j:j+batch_size],theta)
            theta=theta-alpha*temp
        cost[i] = computeCost(X, y, theta) 
    return theta, cost

batch_size=32
alpha = 0.01
iters = 100
start_time = time.time()#start to calculate the time that spent the func whhe we call it
theta_res,cost =mini_batch(X, y, theta, alpha, iters,batch_size)
end_time = time.time()# when the func ends
total_time= end_time - start_time#get the result of the time
time_list=np.matrix([i*total_time/iters for i in range(iters)])#divide thhe time to steps to plot the chart 
fig, ax = plt.subplots()
ax.plot(time_list.T,cost,'r')
ax.set_xlabel('time')
ax.set_ylabel('Cost')
ax.set_title('Cost vs. time')
print("time taken:",total_time)
print(theta_res)#the result of theta vector 

#we can see that mini batch take less time to get optimize vector of theta and with 
# less number of iterations 
# so take part of data every time is more quick from taking all together at ones
"""

"""
def momentum( X, y,theta, alpha,iters,beta):
    cost = np.zeros(iters)
    for i in range(iters):
        theta_v = np.matrix(np.zeros(theta.shape))#vector for v
        theta_v=beta*theta_v+alpha*total_j(X,y,theta)#v=b*v+a*j(theta)
        theta-=theta_v#thheta=thheta-v
        cost[i] = np.mean(np.square(np.dot(X, theta.T) - y))#save the values
    return theta, cost
 
alpha = 0.1
iters = 100
beta=0.9
start_time = time.time()#start to calculate the time that spent the func whhe we call it
theta_res,cost = momentum(X, y, theta, alpha, iters,beta)
end_time = time.time()# when the func ends
total_time= end_time - start_time#get the result of the time
time_list=np.matrix([i*total_time/iters for i in range(iters)])#divide thhe time to steps to plot the chart 
fig, ax = plt.subplots()
ax.plot(time_list.T,cost,'r')
ax.set_xlabel('time')
ax.set_ylabel('Cost')
ax.set_title('Cost vs. time')
print("time taken:",total_time)
print(theta_res)#the result of theta vector 
"""