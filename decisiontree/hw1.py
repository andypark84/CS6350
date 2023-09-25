import numpy as np
import pandas as pd
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None, criterion='gini'):
        self.max_depth = max_depth
        self.criterion = criterion 
        self.tree = None
        self.y = None
        
    def fit(self, X, y, depth=0):
      self.tree = self.build_tree(X, y, depth)
      self.y = y
    
    def build_tree(self, X, y, depth):
      if len(np.unique(y)==1) or (self.max_depth is not None and depth == self.max_depth):
        return Counter(y).most_common(1)[0][0]
        
      best_attribute = self.select_attribute(X,y)
      tree = {best_attribute:{}}
       
      for value in np.unique(X[best_attribute]):
        X_sub = X[X[best_attribute]==value]
        y_sub = y[X[best_attribute]==value]
        tree[best_attribute][value] = self.build_tree(X_sub, y_sub, depth+1)
          
      return tree
    
    def select_attribute(self, X, y):
      if self.criterion == 'entropy':
        return self.entropy_information_gain(X, y)
      elif self.criterion == 'majority_error':
        return self.majority_error_information_gain(X, y)
      elif self.criterion == 'gini':
        return self.gini_index_information_gain(X, y)
      
    def entropy_information_gain(self, X, y):
      entropy_before = self.entropy(y)
      best_attribute = None
      max_IG = 0

      for attribute in X.columns:
        IG = entropy_before - self.entropy_after(X, y, attribute)
        if IG > max_IG:
          max_IG = IG
          best_attribute = attribute

      return best_attribute 

    def entropy(self,y):
      entropy = 0
      
      for label in np.unique(y):
        prob = np.mean(y==label)
        entropy -= prob * np.log2(prob)
      
      return entropy
    
    def entropy_after(self, X, y, attribute):
      conditional_entropy = 0
      
      for value in np.unique(X[attribute]):
        y_sub = y[X[attribute] == value]
        conditional_entropy += (len(y_sub) / len(y)) * self.entropy(y_sub)  
      
      return conditional_entropy

    
    def majority_error_information_gain(self, X, y):
      majority_error_before = self.majority_error(y)
      best_attribute = None
      max_IG = 0

      for attribute in X.columns:
        IG = majority_error_before - self.majority_error_after(X, y, attribute)
        if IG > max_IG:
          max_IG = IG
          best_attribute = attribute

      return best_attribute 

    def majority_error(self,y):
      max_prob = 0
      
      for label in np.unique(y):
        prob = np.mean(y==label)
        if prob > max_prob:
          max_prob = prob
          max_prob
        
        majority_error = 1 - max_prob
      return majority_error
    
    def majority_error_after(self, X, y, attribute):
      conditional_ME = 0
      
      for value in np.unique(X[attribute]):
        y_sub = y[X[attribute] == value]
        conditional_ME += (len(y_sub) / len(y)) * self.majority_error(y_sub)  
      
      return conditional_ME

    def gini_index_information_gain(self, X, y):
        gini_index_before = self.gini_index(y)
        best_attribute = None
        max_IG = 0

        for attribute in X.columns:
          IG = gini_index_before - self.gini_index_after(X, y, attribute)
          if IG > max_IG:
            max_IG = IG
            best_attribute = attribute

        return best_attribute 

    def gini_index(self,y):
      gini_index = 1
      
      for label in np.unique(y):
        prob = np.mean(y==label)
        gini_index -= prob**2
      
      return gini_index
    
    def gini_index_after(self, X, y, attribute):
      conditional_GI = 0
      
      for value in np.unique(X[attribute]):
        y_sub = y[X[attribute] == value]
        conditional_GI += (len(y_sub) / len(y)) * self.gini_index(y_sub)
      
      return conditional_GI

    def predict(self, X):
      predictions = []

      for idx , row in X.iterrows():
          node = self.tree
          while isinstance(node, dict):
              attribute = list(node.keys())[0]
              value = row[attribute]
              node = node[attribute].get(value, None)
          if node is not None:
            predictions.append(node)
          else:
            predictions.append(Counter(self.y).most_common(1)[0][0])
      return predictions

def train_test_split(train,test):
    x_train = train.iloc[:,:-1] 
    y_train = train.iloc[:,-1]
    x_test = test.iloc[:,:-1] 
    y_test = test.iloc[:,-1]
    return x_train, y_train, x_test, y_test

def pred_error(y_true,y_pred):
    acc=0
    for t,p in zip (y_true,y_pred):
        if t==p:
            acc+=1
    return 1-(acc/len(y_true))

def model_pred(i,criterion):
    tree = DecisionTree(max_depth=i, criterion=criterion)
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test) 
    error=pred_error(y_test,y_pred)  
    return error

if __name__ == "__main__":
    car_train = pd.read_csv('./data/car-4/train.csv')
    car_test = pd.read_csv('./data/car-4/test.csv')
    x_train, y_train, x_test, y_test = train_test_split(car_train, car_test)

    criterion=['entropy','majority_error','gini']
    result=[]
    for i in range(1,7):
        ent=model_pred(i,criterion[0])
        ma=model_pred(i,criterion[1])
        gini=model_pred(i,criterion[2])
        result.append([i,ent,ma,gini])
    result=pd.DataFrame(result,columns=['depth',f'{criterion[0]}_pred_error',f'{criterion[1]}_pred_error',f'{criterion[2]}_pred_error'])
    print(result)
    print(f"{criterion[0]} average predict error : ",result[f'{criterion[0]}_pred_error'].mean())
    print(f"{criterion[1]} average predict error : ",result[f'{criterion[1]}_pred_error'].mean())
    print(f"{criterion[2]} average predict error : ",result[f'{criterion[2]}_pred_error'].mean())



