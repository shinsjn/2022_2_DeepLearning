import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

'''
#data load
df_row = pd.read_csv('./midterm_data/new_train_open_col.csv')
print(df_row.head())

#label encoding
le = LabelEncoder()
train_y = le.fit_transform(df_row['Label'])
sample = ['A','B','C']
#print(le.transform(sample))
train_x = df_row.loc[:, df_row.columns!= 'Label']
#print(train_x.head())
#print(train_y.shape)

#data scaling


scaler1 = StandardScaler()
scaler1.fit(train_x)
trans_train_x1 = scaler1.transform(train_x)
#print(scaler.mean_)


model1 = RandomForestClassifier()
model2 = XGBClassifier()

scores1 = cross_val_score(model1, trans_train_x1, train_y, cv=5)
scores2 = cross_val_score(model2, trans_train_x1, train_y, cv=5)
print('Train accuracy 1 :', scores1)
print('Train accuracy 2 :', scores2)
'''




'''
plt.scatter(x = df_row['feature1'],y = df_row['Label'])
plt.title('1')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/1.png')
plt.show()
plt.scatter(x = df_row['feature2'],y = df_row['Label'])
plt.title('2')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/2.png')
plt.show()
plt.scatter(x = df_row['feature3'],y = df_row['Label'])
plt.title('3')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/3.png')
plt.show()
plt.scatter(x = df_row['feature4'],y = df_row['Label'])
plt.title('4')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/4.png')
plt.show()
plt.scatter(x = df_row['feature5'],y = df_row['Label'])
plt.title('5')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/5.png')
plt.show()
plt.scatter(x = df_row['feature6'],y = df_row['Label'])
plt.title('6')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/6.png')
plt.show()
plt.scatter(x = df_row['feature7'],y = df_row['Label'])
plt.title('7')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/7.png')
plt.show()
plt.scatter(x = df_row['feature8'],y = df_row['Label'])
plt.title('8')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/8.png')
plt.show()
plt.scatter(x = df_row['feature9'],y = df_row['Label'])
plt.title('9')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/9.png')
plt.show()
plt.scatter(x = df_row['feature10'],y = df_row['Label'])
plt.title('10')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/10.png')
plt.show()
plt.scatter(x = df_row['feature11'],y = df_row['Label'])
plt.title('11')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/11.png')
plt.show()
plt.scatter(x = df_row['feature12'],y = df_row['Label'])
plt.title('12')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/12.png')
plt.show()
plt.scatter(x = df_row['feature13'],y = df_row['Label'])
plt.title('13')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/13.png')
plt.show()
plt.scatter(x = df_row['feature14'],y = df_row['Label'])
plt.title('14')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/14.png')
plt.show()
plt.scatter(x = df_row['feature15'],y = df_row['Label'])
plt.title('15')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/15.png')
plt.show()
plt.scatter(x = df_row['feature16'],y = df_row['Label'])
plt.title('16')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/16.png')
plt.show()
plt.scatter(x = df_row['feature17'],y = df_row['Label'])
plt.title('17')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/17.png')
plt.show()
plt.scatter(x = df_row['feature18'],y = df_row['Label'])
plt.title('18')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/18.png')
plt.show()
plt.scatter(x = df_row['feature19'],y = df_row['Label'])
plt.title('19')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/19.png')
plt.show()
plt.scatter(x = df_row['feature20'],y = df_row['Label'])
plt.title('20')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/20.png')
plt.show()
plt.scatter(x = df_row['feature21'],y = df_row['Label'])
plt.title('21')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/21.png')
plt.show()
plt.scatter(x = df_row['feature22'],y = df_row['Label'])
plt.title('22')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/22.png')
plt.show()
plt.scatter(x = df_row['feature23'],y = df_row['Label'])
plt.title('23')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/23.png')
plt.show()
plt.scatter(x = df_row['feature24'],y = df_row['Label'])
plt.title('24')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/24.png')
plt.show()
plt.scatter(x = df_row['feature25'],y = df_row['Label'])
plt.title('25')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/25.png')
plt.show()
plt.scatter(x = df_row['feature26'],y = df_row['Label'])
plt.title('26')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/26.png')
plt.show()
plt.scatter(x = df_row['feature27'],y = df_row['Label'])
plt.title('27')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/27.png')
plt.show()
plt.scatter(x = df_row['feature28'],y = df_row['Label'])
plt.title('28')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/28.png')
plt.show()
plt.scatter(x = df_row['feature29'],y = df_row['Label'])
plt.title('29')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/.29png')
plt.show()
plt.scatter(x = df_row['feature30'],y = df_row['Label'])
plt.title('30')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/30.png')
plt.show()
plt.scatter(x = df_row['feature31'],y = df_row['Label'])
plt.title('31')
plt.savefig('C:/Users/shins/PycharmProjects/2022_2_DeepLearning/midterm_data/new_data_anal/31.png')
plt.show()
'''


'''
df_row = pd.read_csv('./midterm_data/new_train_open_col.csv')
print(df_row.head())

#label encoding
le = LabelEncoder()
train_y = le.fit_transform(df_row['Label'])

#print(le.transform(sample))

train_x = df_row.loc[:, df_row.columns!= 'Label']

model = RandomForestClassifier(n_estimators=500, random_state=1234)
# Train the model using the training sets

#model.fit(train_X, train_y)
model.fit(train_x,train_y)

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

for f in range(train_x.shape[1]):
    print("{}. feature {} ({:.3f})".format(f + 1,train_x.columns[indices][f], importances[indices[f]]))

plt.figure()
plt.title('importance')
plt.bar(range(train_x.shape[1]), importances[indices],color="r", yerr = std[indices], align='center')
plt.xticks(range(train_x.shape[1]),train_x.columns[indices], rotation=45)
plt.xlim([-1,train_x.shape[1]])
plt.show()

from xgboost import XGBClassifier
import xgboost

xgb = XGBClassifier(booster='gbtree', importance_type='gain')
xgb.fit(train_x, train_y)
print(xgb.feature_importances_)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20,8))

axes = [ax for row_axes in axes for ax in row_axes]

xgboost.plot_importance(xgb, importance_type='gain', title='gain', xlabel='', grid=False, ax=axes[0])
xgboost.plot_importance(xgb, importance_type='cover', title='cover', xlabel='', grid=False, ax=axes[1])
xgboost.plot_importance(xgb, importance_type='weight', title='weight', xlabel='', grid=False, ax=axes[2])
xgboost.plot_importance(xgb, importance_type='total_gain', title='total_gain', xlabel='', grid=False, ax=axes[3])
xgboost.plot_importance(xgb, importance_type='total_cover', title='total_cover', xlabel='', grid=False, ax=axes[4])
plt.tight_layout()
plt.show()



from sklearn.feature_selection import RFE

model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=10)
fit = rfe.fit(train_x, train_y)

print("Num Features: ",fit.n_features_) 
print("Selected Features: ", fit.support_)
print("Feature Ranking: ",fit.ranking_)



from sklearn.ensemble import ExtraTreesClassifier

etc_model = ExtraTreesClassifier()
etc_model.fit(train_x, train_y)

print(etc_model.feature_importances_)
feature_list = pd.concat([pd.Series(train_x.columns), pd.Series(etc_model.feature_importances_)], axis=1)
feature_list.columns = ['features_name', 'importance']
feature_list.sort_values("importance", ascending =False)[:]
print(feature_list.sort_values("importance", ascending =False)[:])



from sklearn.feature_selection import SelectKBest, f_classif

selectK = SelectKBest(score_func=f_classif)
X = selectK.fit_transform(train_x, train_y)
all_names = train_x.columns
## selector.get_support()
selected_mask = selectK.get_support()
## 선택된 특성(변수)들
selected_names = all_names[selected_mask]
## 선택되지 않은 특성(변수)들
unselected_names = all_names[~selected_mask]
print('Selected names: ', selected_names)
print('Unselected names: ', unselected_names)



df_row = pd.read_csv('./midterm_data/new_train_open_col.csv')

train_x = df_row[df_row.columns.difference(['feature31','feature15','feature30','feature23','feature3','feature9','feature28','feature21','feature27','feature18','feature12','feature6','feature25'])]

sns.heatmap(data=train_x.corr(), fmt = '.2f', linewidths=.32,cmap='Blues')
plt.show()
'''


# Ater feature selection
df_row = pd.read_csv('./midterm_data/new_train_open_col.csv')
print(df_row.head())
test = pd.read_csv('./midterm_data/test_open_col.csv')
#label encoding
le = LabelEncoder()
train_y = le.fit_transform(df_row['Label'])
#print(le.transform(sample))

train_x1 = df_row[df_row.columns.difference(['Label','feature3','feature6','feature9','feature12','feature15','feature18','feature21','feature23','feature25','feature27','feature28','feature30','feature31'])]
test_x1 = test[test.columns.difference(['feature3','feature6','feature9','feature12','feature15','feature18','feature21','feature23','feature25','feature27','feature28','feature30','feature31'])]

#train_x1 = df_row[['feature7','feature8','feature11','feature13','feature14','feature16','feature17','feature19','feature20','feature24']]
#test_x1 = test[['feature7','feature8','feature11','feature13','feature14','feature16','feature17','feature19','feature20','feature24']]

#train_x1 = df_row.loc[:, df_row.columns!= 'Label']
#test_x1=test
#print(train_x.head())
#print(train_y.shape)

#data scaling

scaler1 = StandardScaler()
scaler1.fit(train_x1)
trans_train_x1 = scaler1.transform(train_x1)
test_x1 = scaler1.transform(test_x1)

#scaler2 = StandardScaler()
#scaler2.fit(train_x2)
#trans_train_x2 = scaler2.transform(train_x2)
#print(scaler.mean_)

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from itertools import combinations

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'''
X_train, X_test, y_train, y_test = train_test_split(trans_train_x1,train_y,test_size=0.2)

xgb = XGBClassifier(booster= 'gbtree', gamma= 0, learning_rate= 0.2,max_depth= 6, n_estimators= 200, tree_method= 'exact')
gbc = GradientBoostingClassifier(learning_rate= 0.15, n_estimators= 400)
knn1 = KNeighborsClassifier(metric= 'manhattan', n_neighbors= 5, weights= 'distance')
knn2 = KNeighborsClassifier(metric= 'manhattan', n_neighbors= 6, weights= 'distance')
lgbm = LGBMClassifier(learning_rate= 0.2, max_depth= -1, min_child_samples= 10, num_leaves= 40, reg_alpha= 0)
rf = RandomForestClassifier(max_depth= 12, min_samples_leaf= 10, min_samples_split= 10, n_estimators= 100)
svc = SVC(C= 10.0, gamma= 0.1, kernel= 'rbf',probability=True)
estimator_list = [('xgb',xgb),('gbc',gbc),('knn1',knn1),('knn2',knn2),('lgbm',lgbm),('rf',rf),('svc',svc)]
c_num = [1,2,3,4,5,6,7]

best_acc = 0
best_list = []

second_acc = 0
second_list=[]

third_acc = 0
third_list=[]


for num in c_num:
    sample = list(combinations(estimator_list,num))
    for estimators in sample:
        temp_list = []
        voting_model = VotingClassifier(estimators=estimators, voting='soft')
        voting_model.fit(X_train, y_train)
        pred = voting_model.predict(X_test)
        print('==========')
        for x in estimators:
            print(x[0])
            temp_list.append(x[0])
        acc = accuracy_score(y_test, pred)
        print('보팅 분류기의 정확도: {0: .4f}'.format(accuracy_score(y_test, pred)))
        print('==========')
        if acc>third_acc:
            if acc>second_acc:
                if acc>best_acc:
                    best_acc=acc
                    best_list = temp_list

                else:
                    second_acc=acc
                    second_list=temp_list
            else:
                third_acc=acc
                third_list=temp_list

print(best_list)
print(best_acc)

print(second_list)
print(second_acc)

print(third_list)
print(third_acc)
'''

xgb = XGBClassifier(booster= 'dart', gamma= 0, learning_rate= 0.2,max_depth= 6, n_estimators= 200, tree_method= 'exact')
gbc = GradientBoostingClassifier(learning_rate= 0.15, n_estimators= 400)
knn1 = KNeighborsClassifier(metric= 'manhattan', n_neighbors= 5, weights= 'distance')
knn2 = KNeighborsClassifier(metric= 'manhattan', n_neighbors= 6, weights= 'distance')
lgbm = LGBMClassifier(learning_rate= 0.2, max_depth= -1, min_child_samples= 10, num_leaves= 40, reg_alpha= 0)
rf = RandomForestClassifier(max_depth= 12, min_samples_leaf= 10, min_samples_split= 10, n_estimators= 100)
svc = SVC(C= 10.0, gamma= 0.1, kernel= 'rbf',probability=True)

estimator1 = [('xgb',xgb),('knn2',knn2),('rf',rf),('svc',svc)]
estimator2 = [('xgb',xgb),('knn1',knn1),('rf',rf),('svc',svc)]
estimator3 = [('xgb',xgb),('knn2',knn2),('svc',svc)]
estimator4 = [('xgb',xgb),('knn2',knn2),('svc',svc)]



X_train, X_test, y_train, y_test = train_test_split(trans_train_x1,train_y,test_size=0.2)


voting_model = VotingClassifier(estimators=estimator1, voting='soft')
voting_model.fit(trans_train_x1, train_y)
pred1 = voting_model.predict(X_test)
acc1 = accuracy_score(y_test, pred1)
print("astimator1:",acc1)

pred1 = voting_model.predict(test_x1)
pred1 = le.inverse_transform(pred1)
DF1 = pd.DataFrame(pred1)
print(pred1)
DF1.to_csv('./estimator1.csv',mode='w')

voting_model = VotingClassifier(estimators=estimator2, voting='soft')
voting_model.fit(trans_train_x1, train_y)
pred2 = voting_model.predict(X_test)
acc2 = accuracy_score(y_test, pred2)
print("astimator2:",acc2)

pred2 = voting_model.predict(test_x1)
pred2 = le.inverse_transform(pred2)

DF2 = pd.DataFrame(pred2)
DF2.to_csv('./estimator2.csv',mode='w')

voting_model = VotingClassifier(estimators=estimator3, voting='soft')
voting_model.fit(trans_train_x1, train_y)
pred3 = voting_model.predict(X_test)
acc3 = accuracy_score(y_test, pred3)
print("astimator3:",acc3)

pred3 = voting_model.predict(test_x1)
pred3 = le.inverse_transform(pred3)
DF3 = pd.DataFrame(pred3)
DF3.to_csv('./estimator3.csv',mode='w')

voting_model = VotingClassifier(estimators=estimator4, voting='soft')
voting_model.fit(trans_train_x1, train_y)
pred4 = voting_model.predict(X_test)
acc4 = accuracy_score(y_test, pred4)
print("astimator4:",acc4)

pred4 = voting_model.predict(test_x1)
pred4 = le.inverse_transform(pred4)
DF4 = pd.DataFrame(pred4)
DF4.to_csv('./estimator4.csv',mode='w')





'''
scores1 = cross_val_score(voting_model, trans_train_x1, train_y, cv=5)
print('Train accuracy 1 di :', scores1)


model1 = RandomForestClassifier()
model2 = XGBClassifier()
scores1 = cross_val_score(model1, trans_train_x1, train_y, cv=5)
scores2 = cross_val_score(model2, trans_train_x1, train_y, cv=5)
print('Train accuracy 1 di :', scores1)
print('Train accuracy 2 di :', scores2)
'''

'''
#model1 = RandomForestClassifier()
#model2 = XGBClassifier()
#scores1 = cross_val_score(model1, trans_train_x2, train_y, cv=5)
#scores2 = cross_val_score(model2, trans_train_x2, train_y, cv=5)
#print('Train accuracy 1 di :', scores1)
#print('Train accuracy 2 di :', scores2)
'''


'''
param_grid_XGB = [
    {
    'booster':['gbtree','dart'],
    'learning_rate':[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45],
     'max_depth':[5,6,7,8,9],
     'colsample_bytree':[0.5,0.8,0.9,1],
     'reg_lambda':[0,0.8,0.9,1],
     'reg_alpha':[0,0.1,0.3,0.5,1,2,3],
     'gamma':[0,0.3,0.5,1,2],
        'min_child_weight':[1,2,3],
        'n_estimators':[100,200,300]
    }
  ]
model = XGBClassifier(random_state=42,eval_metric='mlogloss',use_label_encoder=False,objective='mulit:sofprob')
grid_search = RandomizedSearchCV(model, param_grid_XGB, cv=5,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10,n_iter=1000)
'''
'''
param_grid_KNN = [
    {
    'n_neighbors' : list(range(1,50)),
    'weights' : ["uniform", "distance"],
    'metric' : ['euclidean', 'manhattan', 'minkowski']
    }
  ]
'''
'''
from sklearn.svm import SVC
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid_Logistic_svc = [{'C': param_range,
               'kernel': ['linear']},
              {'C': param_range,
               'gamma': param_range,
               'kernel': ['rbf']}]
'''
'''
param_grid_LR = [
    {
    'penalty':['elasticnet'],
    'l1_ratio':[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
     'C':[1e-3,1e-2,1e-1,1.0,10],
     'solver':['newton-cg','lbfgs','liblinear','sag','saga'],
     'max_iter':[200,400,600],
     'multi_class':['ovr','multinomial']
    }
  ]
from sklearn.linear_model import LogisticRegression
'''
'''
parameters = {'max_depth': [3, 5, 7,10],
              'min_samples_split': [3, 5,7,10],
              'splitter': ['best', 'random'],
              'criterion': ['gini','entropy'],
              'max_features':['int', 'float', 'auto', 'sqrt', 'log2','None']}

from sklearn.tree import DecisionTreeClassifier

'''

'''
params = { 'n_estimators' : [10, 100,200,400],
           'max_depth' : [6, 8, 10, 12],
           'min_samples_leaf' : [10,20,30],
           'min_samples_split' : [10,20,30]
            }
'''
'''
params = {
    'n_estimators':[50,100,200,400],
     'learning_rate':[0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6]
         }
from sklearn.ensemble import GradientBoostingClassifier
'''
'''
parameters = {'num_leaves':[20,40,60,80,100], 'min_child_samples':[5,10,15],'max_depth':[-1,5,10,20],
             'learning_rate':[0.05,0.1,0.2],'reg_alpha':[0,0.01,0.03]}
from lightgbm import LGBMClassifier
'''
'''
param_grid_XGB = [
    {
    'booster':['gbtree','dart'],
        'tree_method':['exact'],
    'learning_rate':[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45],
     'max_depth':[6,7,8,9,10],
     'gamma':[0,0.3,0.5,1,2],
        'n_estimators':[100,200,300],
    }
  ]
model = XGBClassifier(use_label_encoder =False,eval_metric='mlogloss')
grid_search = GridSearchCV(model, param_grid_XGB, cv=5,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)

grid_search.fit(trans_train_x1, train_y)
print(grid_search.best_score_)
print(grid_search.best_params_)

scores = pd.DataFrame(grid_search.cv_results_)
print(scores)
scores.to_csv('./paramResult_XGB_NEW.csv')
'''

'''
# performance evaluation
print('Train accuracy :', model.score(train_X, train_y))
print('Test accuracy :', model.score(test_X, test_y))
pred_y= model.predict(test_X)
confusion_matrix(test_y, pred_y)
'''
