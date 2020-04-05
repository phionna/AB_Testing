#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math


# In[2]:


df = pd.read_csv('AB_test_data.csv')


# In[3]:


df.head()


# # Part 1 - Hypo Test

# In[5]:


df_A = df[df['Variant'] == 'A']
df_B = df[df['Variant'] == 'B']

A_conv_rate = len(df_A[df_A['purchase_TF'] == True]) / len(df_A)
B_conv_rate = len(df_B[df_B['purchase_TF'] == True]) / len(df_B)
n = len(df_B)


# Null Hypothesis: B_conv_rate = A_conv_rate
# 
# Alternative Hypothesis: B_conv_rate >= A_conv_rate

# In[6]:


#Calculate Z-score
z = (B_conv_rate - A_conv_rate) / math.sqrt((A_conv_rate * (1- A_conv_rate))/n)
print(z)


# Z (alpha) = 1.64
# 
# Hence we reject the null hypothesis and conclude that B conversion rate does indeed have significantly higher conversion rate than A.

# # Part 2 - Optimal Sample Size

# In[7]:


t_alpha = 1.96
p_bar = (A_conv_rate + B_conv_rate) / 2
p0 = A_conv_rate
p1 = B_conv_rate
delta = (B_conv_rate - A_conv_rate)
t_beta = 0.842

optimal = (t_alpha * math.sqrt((2*p_bar*(1-p_bar))) + t_beta * math.sqrt(p0*(1-p0) + p1*(1-p1)))**2 * (1/(delta**2))
print(optimal)


# ## Sampling Optimal Sizes from Data

# Conduct the test 10 times using samples of the optimal size. Report results.

# In[9]:


list_of_z_scores = []

for i in range(10):
    sample = df_B.sample(n=1157,axis=0)
    B_conv_rate = len(sample[sample['purchase_TF'] == True]) / len(sample)
    
    z = (B_conv_rate - A_conv_rate) / math.sqrt((A_conv_rate * (1- A_conv_rate))/len(sample))
    list_of_z_scores.append(z)

list_of_z_scores


# In[10]:


list_of_success = []

for i in list_of_z_scores:
    if i > 1.96:
        list_of_success.append(True)
    else:
        list_of_success.append(False)


# In[11]:


sum(list_of_success)/len(list_of_success)


# # Part 3-  Conduct Sequential Test

# Conduct a sequential test for the 10 samples. For any of the samples, were you able to stop the test prior to using the full sample? What was the average number of iterations required to stop the test?

# Under H0: P(x=1) = 0.15206
# 
# Under H1: P(x=1) = 0.1962
# 
# Type I error: 5%
# 
# Type II error: 20%

# In[74]:


upper = np.log(1/0.05)
lower = np.log(0.2)

p0 = 0.15206
p1 = 0.1962


# In[236]:


def feed_criteria(total_criteria,sample,number):
    global len_log
    global success_log
    
    if (total_criteria <= lower):
        print("test stopped and accept H0")
        print(len(log))
        len_log.append(len(log))
        success_log.append(0)
    elif total_criteria >= upper:
        print('test stopped and accept H1')
        print(len(log))
        len_log.append(len(log))
        success_log.append(1)

    else:
        #print("keep going")
        if sample.purchase_TF.iloc[number] == True:
            criteria = np.log(p1/p0)
        else:
            criteria = np.log((1-p1)/(1-p0))
        log.append(criteria)
        end_criteria = sum(log)
        number = number +1
        #print("current number:",end_criteria)
        feed_criteria(end_criteria,sample,number)


# In[240]:


len_log = []
success_log = []
for i in range(100000):
    sample = df_B.sample(n=1157,axis=0)
    number = 0
    log = []
    feed_criteria(0,sample,number)


# In[241]:


#Avg number of iterations
sum(len_log)/len(len_log)


# In[242]:


#Avg number of successes
sum(success_log)/len(success_log)

