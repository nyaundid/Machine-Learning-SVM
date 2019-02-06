
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn import svm

# Packages for visuals
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

# Allows charts to appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Pickle package
import pickle
from matplotlib import style
style.use("ggplot")

import time
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import linear_model

from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os 

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score


# Packages for analysis# Packag 
import pandas as pd
import numpy as np
from sklearn import svm

# Packages for visuals
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

# Allows charts to appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Pickle package
import pickle


# In[3]:


parser = lambda date: pd.to_datetime(date, format='%d.%m.%Y')
userhome = os.path.expanduser('~')
path = '/Documents/datav2/dataversion2/'
X_train = pd.read_csv(userhome + path + 'X_train1.csv')
Y_train = pd.read_csv(userhome + path + 'Y_train1.csv',header=None)
X_test = pd.read_csv(userhome + path + 'X_test1.csv')
Y_test = pd.read_csv(userhome + path + 'Y_test1.csv',header=None)


# In[4]:


X_train


# In[28]:


Y_train


# In[5]:


recipe_features = X_train.columns.values[1:].tolist()
recipe_features


# In[6]:


X1 = X_train.loc[:, ["Item Type_Electronics"]]


# In[7]:


X1


# In[8]:


electronics= X1.loc[X1["Item Type_Electronics"] == 1]


# In[9]:


electronics


# In[10]:


nonelectronics= X1.loc[X1["Item Type_Electronics"] == 0]


# In[11]:


nonelectronics


# In[12]:


all = pd.concat([X_train,Y_train],axis=1)


# In[13]:


all


# In[14]:


all2 = pd.concat([X_train,Y_train],axis=1)


# In[15]:


from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report


# In[16]:


from sklearn.model_selection import RandomizedSearchCV


# In[17]:


from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV



# In[ ]:


from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 50, 100, 300, 500, 800, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, Y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()


# In[18]:


# Tuning hyper-parameters for precision

clf = svm.SVC(kernel = 'linear', C = 300, probability = True ).fit(X_train, Y_train)
y_predicted = clf.predict(X_test)


# In[90]:


from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print('confusion_matrix')
print(confusion_matrix(Y_test,y_predicted))
print()
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print()
print("The scores are computed on the full evaluation set.")
y_predicted = clf.predict(X_test)
print()
print(clf.score(X_test, Y_test))
print()
print(classification_report(Y_test, y_predicted ))
print()
print()


# In[ ]:




1 q1 q1 
0 q3 q4


# In[91]:


import numpy as np
from sklearn import metrics

y_pred_proba2 = clf.predict_proba(X_test)[::,0]
auc = metrics.roc_auc_score(Y_test, y_pred_proba2)
fpr, tpr, thresholds = metrics.roc_curve(Y_test,y_pred_proba2,pos_label=0)

plt.figure()
lw = 2
plt.plot(fpr,tpr,label="data 1")
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[92]:


import numpy as np
from sklearn import metrics



y_pred_proba1 = clf.predict_proba(X_test)[::,1]

fpr, tpr, thresholds = metrics.roc_curve(Y_test,y_pred_proba1, pos_label=1)



plt.figure()
lw = 2
plt.plot(fpr,tpr,label="data 1")
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[93]:


y_pred_proba1 = clf.predict_proba(X_test)[::,1]

fpr2, tpr2, thresholds = metrics.roc_curve(Y_test,y_pred_proba1,pos_label=1)


y_pred_proba = clf.predict_proba(X_test)[::,0]

fpr, tpr, thresholds = metrics.roc_curve(Y_test,y_pred_proba, pos_label=0)
plt.figure()
lw = 2
plt.plot(fpr,tpr,label="data 1")
plt.plot(fpr2,tpr2)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for both')
plt.legend(loc="lower right")
plt.show()


# In[94]:


y_pred_proba = clf.predict_proba(X_test)[::,1]
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
auc


# In[95]:


y_pred_proba = clf.predict_proba(X_test)[::,0]
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
auc


# In[96]:


from sklearn.metrics import roc_curve,roc_auc_score,auc

 
pred_0 = clf.predict_proba(X_test)[:,0]
pred_1 = clf.predict_proba(X_test)[:,1]

 
 
# Compute ROC curve and ROC area for each class
n_classes = 2
fpr_0, tpr_0, threshold_0 = roc_curve(Y_test, pred_0,0)
roc_auc_0 = auc(fpr_0, tpr_0)
fpr_1, tpr_1, threshold_1 = roc_curve(Y_test, pred_1,1)
roc_auc_1 = auc(fpr_1, tpr_1)

 
 
plt.figure()
lw = 2
fig, ax = plt.subplots(figsize=(18,8))

plt.plot(fpr_0, tpr_0, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_0)
plt.plot(fpr_1, tpr_1, color='green',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_1)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

