#!/usr/bin/env python
# coding: utf-8

# In[1]:


#有些时候 我们需要通过相同的feature来预测多个目标，这个时候就需要使用MultiOutputRegressor包来进行多回归
import tensorflow 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.multioutput import MultiOutputRegressor  
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
import xgboost
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import joblib
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import datetime


# In[2]:


X = pd.read_csv('E:\\进展\\滑坡预测\\input20.csv',header=None)
X = X.values
X.shape
y = pd.read_csv('E:\\进展\\滑坡预测\\output20.csv',header=None)
y = y.values
y.shape
seed = 7
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=seed)
all_x_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X_train = all_x_scaler.fit_transform(X_train)
X_test = all_x_scaler.transform(X_test)


# In[3]:


font2 = {'family':'Times New Roman','weight': 'normal','size': 30}
font1 = {'family':'Times New Roman','weight': 'bold','size': 30}
font3 ={'family':'SimSun','weight': 'bold','size': 30}
font6 ={'family':'Microsoft YaHei','weight': 'normal','size': 10}


# In[4]:


xgb = XGBClassifier(max_depth=9,
                    min_child_weight=2,
                     n_estimators=1000,
                     n_jobs=4,
                     seed=888,tree_method='gpu_hist',
                   )
xgb.fit(X_train,y_train,eval_metric=['merror','auc', 'mlogloss'],early_stopping_rounds=10, eval_set=[[X_train, y_train], [X_test, y_test]], verbose=True )
score=xgb.score(X_test,y_test)
print("xgb的准确率为%f"%score)
current_time = datetime.datetime.now()
print("current_time:    " + str(current_time))


# In[5]:


results = xgb.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)

# Create lists to store data for log loss and classification error
train_log_loss = results['validation_0']['mlogloss']
test_log_loss = results['validation_1']['mlogloss']
train_error = results['validation_0']['merror']
test_error = results['validation_1']['merror']

# Create a DataFrame with the collected data
df = pd.DataFrame({'n_trees': x_axis,
                   'train_log_loss': train_log_loss,
                   'test_log_loss': test_log_loss,
                   'train_error': train_error,
                   'test_error': test_error})

# Save the DataFrame to a CSV file
df.to_csv('D:\\xgboost_evaluation_data_20.csv', index=False)


# In[65]:


results = xgb.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train_log_loss')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test_log_loss')
ax.tick_params(axis='x',direction='in',top=True,right=True)
ax.tick_params(axis='y',direction='in',top=True,right=True)
ax.legend(prop=font1)
plt.xlim([0.0, 600])
plt.ylim([0.0, 3])
plt.yticks(np.arange(0.5,3.5,0.5),fontproperties='Times New Roman', size=15)#设置大小及加粗
plt.xticks(np.arange(0,700,100),fontproperties='Times New Roman', size=15)
ax.legend(prop=font2)
plt.ylabel('Log Loss',font1)
plt.xlabel("n_trees",font1)
plt.title('XGBoost Log Loss',font1)
plt.tight_layout()
plt.savefig("D:\\loss20.tiff")
plt.show()
plt.close()
# plot classification error
fig, ax = plt.subplots()
ax.tick_params(axis='x',direction='in',top=True,right=True)
ax.tick_params(axis='y',direction='in',top=True,right=True)
ax.plot(x_axis, results['validation_0']['merror'], label='Train_error')
ax.plot(x_axis, results['validation_1']['merror'], label='Test_error')
plt.xlim([0.0, 600])
plt.ylim([0.0, 0.8])
plt.yticks(np.arange(0.1,0.9,0.1),fontproperties='Times New Roman', size=15)#设置大小及加粗
plt.xticks(np.arange(0,700,100),fontproperties='Times New Roman', size=15)
ax.legend(prop=font2)
plt.ylabel('Classification Error',font1)
plt.xlabel("n_trees",font1)
plt.title('XGBoost Classification Error',font1)
plt.tight_layout()
plt.savefig("D:\\error20.tiff")
plt.show() 
plt.close()


# In[43]:


y_q=xgb.predict(X_test)

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score
#print('预测数据的Cohen’s Kappa系数为：',
#      cohen_kappa_score(y_test_1,y_q))
from sklearn.metrics import classification_report
# 打印分类报告
print('预测数据的分类报告为：')
print(classification_report(y_test,y_q,digits=4))


# In[44]:


rt=classification_report(y_test,y_q,digits=4,output_dict=True)
t=pd.DataFrame(rt).transpose()
t.to_csv('D:\\classification_report_20.csv')


# In[45]:


ye = xgb.predict(X_test)
s =0
for i in range(186000):
    if abs(ye[i]-y_test[i])<2:
        s=s+1
print(s/186000)


# In[6]:


yt=np.zeros((31000,60))
for i in range(31000):
    for j in range(60):
        yt[i,j]=y[31000*j+i]
X_=all_x_scaler.transform(X)
y_qq=xgb.predict(X_)
y_p = np.zeros((y_qq.shape[0],1))
y_pred = np.zeros((int(y_qq.shape[0]/60),60))
for i in range(31000):
    for j in range(60):
        y_pred[i,j]=y_qq[31000*j+i]
for i in range(y_pred.shape[0]):
    for j in range(y_pred.shape[1]-1):
        if y_pred[i,j+1]<y_pred[i,j]:
            y_pred[i,j+1] = y_pred[i,j]
            
for i in range(31000):
    for j in range(60):
        y_p[31000*j+i]=y_pred[i,j]


# In[7]:


df1 = pd.DataFrame(y_pred)
df2 = pd.DataFrame(yt)
# Save the DataFrame to a CSV file
df1.to_csv('D:\\y_pred_20.csv', index=False)
df1.to_csv('D:\\y_test_20.csv', index=False)


# In[54]:


for i in range(100):
    plt.figure(figsize=(10,6),dpi=100)
    plt.tick_params(axis='x',direction='in',top=True,right=True)
    plt.tick_params(axis='y',direction='in',top=True,right=True)
    x = np.arange(0.5, 30.5, 0.5)
    print(i)
    plt.scatter(x,y_pred[i],color='none',marker='o' ,edgecolors='black',s=30,label='y_pred', clip_on=False)
    #plt.scatter(x,yt[i])
    #plt.plot(x,y_pred[i],label='y_pred',linestyle='--',color='g')
    plt.plot(x,yt[i],"b:",label='y_test',)
    plt.xlim([0, 31])
    plt.ylim([0, yt[i,59]+1])
    plt.yticks(np.arange(0,yt[i,59]+1,5),fontproperties='Times New Roman', size=30)#设置大小及加粗
    plt.xticks([1,5,10,15,20,25,30],fontproperties='Times New Roman', size=30)
    plt.tick_params(labelsize=15)
    plt.xlabel('Elapsed time(days)',font3)
    plt.ylabel('20% Dissociation front(m)',font3)
    plt.legend(loc="lower right",prop=font2)
    plt.show()
    plt.close()


# In[67]:


l=[0,3,4,5,6,9,10,33,34]
for i in l:
    plt.figure(figsize=(10,6),dpi=100)
    plt.tick_params(axis='x',direction='in',top=True,right=True)
    plt.tick_params(axis='y',direction='in',top=True,right=True)
    x = np.arange(0.5, 30.5, 0.5)
    print(i)
    plt.scatter(x,y_pred[i],color='none',marker='o' ,edgecolors='black',s=30,label='y_pred', clip_on=False)
    #plt.scatter(x,yt[i])
    #plt.plot(x,y_pred[i],label='y_pred',linestyle='--',color='g')
    plt.plot(x,yt[i],"b:",label='y_test',)
    plt.xlim([0, 31])
    plt.ylim([0, yt[i,59]+1])
    plt.yticks(np.arange(0,yt[i,59]+1,5),fontproperties='Times New Roman', size=30)#设置大小及加粗
    plt.xticks([1,5,10,15,20,25,30],fontproperties='Times New Roman', size=30)
    plt.tick_params(labelsize=30)
    plt.xlabel('Elapsed time(days)',font1)
    plt.ylabel('20% Dissociation front(m)',font1)
    plt.legend(loc="lower right",prop=font2)
    plt.tight_layout()
    plt.savefig("D:\\20%d.tiff"%i)
    plt.show()
    plt.close()


# In[ ]:


plt.savefig("filename.png")


# In[ ]:





# In[15]:


from xgboost import plot_importance
plot_importance(xgb)
plt.show()


# In[5]:


importance = xgb.feature_importances_

# 创建一个带有特征索引的字典
feature_indices = {i: importance[i] for i in range(len(importance))}

# 对特征重要度进行排序
sorted_indices = sorted(feature_indices, key=feature_indices.get, reverse=True)

# 选择排名前10的特征
top_10_features = sorted_indices[:12]

print("排名前10的特征索引:", top_10_features)


# In[9]:


importance = xgb.get_booster().get_score(importance_type='weight')

# 将特征重要度排序并选择前10个
top_features = sorted(importance, key=importance.get, reverse=True)[1:11]
f_m = ['Time','Residual water saturation','Subloading ratio evolution','Initial porosity','Stress ratio at the critical state',
       'Stiffness enhancement by hydrate','Well pressure','Van genuchten parameter c','Ideal gas saturation','Hydrate number']
f_m2 = ['']
# 仅保留前10个特征的重要度
top_importance = {feature: importance[feature] for feature in top_features}
df = pd.DataFrame({'Feature': list(top_importance.keys()), 'Importance': list(top_importance.values())})
# Save the DataFrame to a CSV file
#df.to_csv('D:\\top_feature_importance_20.csv', index=False)


# In[27]:


importance = xgb.get_booster().get_score(importance_type='weight')

# 将特征重要度排序并选择前10个
top_features = sorted(importance, key=importance.get, reverse=True)[1:11]
top_importance = {feature: importance[feature] for feature in top_features}
print(top_importance)
# 绘制特征重要度图
plt.figure(figsize=(6, 6))
plt.tick_params(axis='x',direction='in',top=True,right=True)
plt.tick_params(axis='y',direction='in',top=True,right=True)
plt.barh(list(top_importance.keys()),top_importance.values(),color='black')
plt.ylabel('Features',font1)
plt.xlabel('Importance',font1)

plt.yticks(fontproperties='Times New Roman', size=25)
plt.xticks(np.arange(0,180000,40000),rotation=0,fontproperties='Times New Roman', size=20)
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
plt.tight_layout()
plt.savefig("D:\\20_importance_.tiff")
plt.show()


# In[58]:


top_importance.keys()


# In[ ]:




