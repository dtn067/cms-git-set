#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import datetime
from functools import reduce
import os

import pandas as pd
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'nbagg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()


# In[2]:


# Data collected from a spark query at CERN, in pandas pickle format
# CRAB jobs only have data after Oct. 2017
ws = pd.read_pickle("data/working_set_day.pkl.gz")
# spark returns lists, we want to use sets
ws['working_set_blocks'] = ws.apply(lambda x: set(x.working_set_blocks), 'columns')
ws['working_set'] = ws.apply(lambda x: set(x.working_set), 'columns')


# In[3]:


#   DBS BLOCKS table schema:
#     BLOCK_ID NOT NULL NUMBER(38)
#     BLOCK_NAME NOT NULL VARCHAR2(500)
#     DATASET_ID NOT NULL NUMBER(38)
#     OPEN_FOR_WRITING NOT NULL NUMBER(38)
#     ORIGIN_SITE_NAME NOT NULL VARCHAR2(100)
#     BLOCK_SIZE NUMBER(38)
#     FILE_COUNT NUMBER(38)
#     CREATION_DATE NUMBER(38)
#     CREATE_BY VARCHAR2(500)
#     LAST_MODIFICATION_DATE NUMBER(38)
#     LAST_MODIFIED_BY VARCHAR2(500)
if not os.path.exists('data/block_size.npy'):
    blocksize = pd.read_csv("data/dbs_blocks.csv.gz", dtype='i8', usecols=(0,5), names=['block_id', 'block_size'])
    np.save('data/block_size.npy', blocksize.values)
    blocksize = blocksize.values
else:
    blocksize = np.load('data/block_size.npy')

# We'll be accessing randomly, make a dictionary
blocksize = {v[0]:v[1] for v in blocksize}


# In[4]:


# join the data tier definitions
datatiers = pd.read_csv('data/dbs_datatiers.csv').set_index('id')
ws['data_tier'] = datatiers.loc[ws.d_data_tier_id].data_tier.values


# In[5]:


date_index = np.arange(np.min(ws.day.values//86400), np.max(ws.day.values//86400)+1)
date_index_ts = np.array(list(datetime.date.fromtimestamp(day*86400) for day in date_index))


# In[6]:


ws_filtered = ws[(ws.crab_job==True) & (ws.data_tier.str.contains('MINIAOD'))]

blocks_day = []
for i, day in enumerate(date_index):
    today = (ws_filtered.day==day*86400)
    blocks_day.append(reduce(lambda a,b: a.union(b), ws_filtered[today].working_set_blocks, set()))

print("Done assembling blocklists")

nrecords = np.zeros_like(date_index)
lifetimes = {
    '1w': 7,
    '1m': 30,
    '3m': 90,
    '6m': 120,
}
ws_size = {k: np.zeros_like(date_index) for k in lifetimes}
nrecalls = {k: np.zeros_like(date_index) for k in lifetimes}
recall_size = {k: np.zeros_like(date_index) for k in lifetimes}
previous = {k: set() for k in lifetimes}

for i, day in enumerate(date_index):
    nrecords[i] = ws_filtered[(ws_filtered.day==day*86400)].size
    for key in lifetimes:
        current = reduce(lambda a,b: a.union(b), blocks_day[max(0,i-lifetimes[key]):i+1], set())
        recall = current - previous[key]
        nrecalls[key][i] = len(recall)
        ws_size[key][i] = sum(blocksize[bid] for bid in current)
        recall_size[key][i] = sum(blocksize[bid] for bid in recall)
        previous[key] = current
    if i%30==0:
        print("Day ", i)

print("Done")


# In[7]:


fig, ax = plt.subplots(1,1)
ax.plot(date_index_ts, recall_size['1w']/1e15, label='1 week')
ax.plot(date_index_ts, recall_size['1m']/1e15, label='1 month')
ax.plot(date_index_ts, recall_size['3m']/1e15, label='3 months')
ax.legend(title='Block lifetime')
ax.set_title('Simulated block recalls for CRAB users')
ax.set_ylabel('Recall rate [PB/day]')
ax.set_xlabel('Date')
ax.set_ylim(0, None)
ax.set_xlim(datetime.date(2017,10,1), None)


# In[27]:


fig, ax = plt.subplots(1,1)
ax.plot(date_index_ts, ws_size['1w']/1e15, label='1 week')
ax.plot(date_index_ts, ws_size['1m']/1e15, label='1 month')
ax.plot(date_index_ts, ws_size['3m']/1e15, label='3 months')
ax.legend(title='Block lifetime')
ax.set_title('Working set for CRAB users, MINIAOD*')
ax.set_ylabel('Working set size [PB]')
ax.set_xlabel('Date')
ax.set_ylim(0, None)
ax.set_xlim(datetime.date(2017,10,1), None)


# In[9]:


recall_size['3m'].mean()/1e12


# In[10]:


print(ws_filtered)


# In[11]:


# block_dict is a dictionary that holds the lists of blocks
# for all of the days for which the lists are nonzero
block_dict = {}
i=0
for el in blocks_day:
    i=i+1
    if len(el)>0:
        block_dict[i] = el

print("Merging daily block lists into one block set")
block_list = []
for i in range(len(blocks_day)):
    block_list += blocks_day[i]
# block_set is a set of all unique blocks.
# This can be used to isolate properties of individual blocks
# (e.g. how many times a block is accessed)
block_set = set(block_list)
print("Block Set Created")


# In[12]:


# block_m_access is a set of blocks that have been accessed 
# in more than one day
block_m_access = set()

first = False
second = False
for day in block_dict:
    if first is False:
        block_m_access.update(block_set.intersection(block_dict[day]))
        first = True
    elif (second is False):
        block_m_access = (block_m_access.intersection(block_dict[day]))
        second = True
    else:
        block_m_access.update(block_m_access.intersection(block_dict[day]))
print("Done")


# In[13]:


print(block_m_access)


# In[14]:


# Initializes all of the access lists and timers
# "timers" are dictionaries with unique block keys and values that are
# lists

# block_access_timer
# keys are unique blocks
# values are the lists of times between first access and the subsequent accesses
block_access_timer = {}
for setBlock in block_set:
    block_access_timer[setBlock] = []
    
# block_first_access
# keys are unique blocks
# values are the time of first access
block_first_access = {}
for setBlock in block_set:
    block_first_access[setBlock] = 0
    
# block_sub_access
# keys are unique blocks
# values are the time of first access
block_sub_access = {}
for setBlock in block_set:
    block_sub_access[setBlock] = 0
    
# block_abs_access
# keys are unique blocks
# values are the day of access
block_abs_access_timer = {}
for setBlock in block_set:
    block_abs_access_timer[setBlock] = []


# In[15]:


# Populates the dictionary, block_access_timer, with the time difference between the
# first access and each subsequent access
def accessTime(block, day):
    if (block_first_access[block] == 0):
        # This stores the inital time (time at which the block was first accessed)
        block_first_access[block] = day
    elif (block_first_access[block] != 0):
        block_sub_access[block] = day
        # Holds the time difference between the first access and each subsequent access
        block_access_timer[block].append(block_sub_access[block] - block_first_access[block])
    else:
        return
    
# Populates the dictionary, block_abs_access, with the day at which the block was 
# accessed
def absAccessTime(block, day):
    block_abs_access_timer[block].append(day)
    
# Removes all of the blocks that did not repeat
def removeTimerRedundancies(timer):
    for block in list(timer.keys()):
        if not timer[block]:
            timer.pop(block)
        
# Iterates over each day and for each block appends to its corresponding list every
# day for which it is accessed
first = False
block_t_access = set()
for day in block_dict:
    if first is False:
        block_t_access.update(block_set.intersection(block_dict[day]))
        first = True
        for block in block_t_access:
            accessTime(block, day)
            absAccessTime(block, day)
    else:
        block_t_access = (block_m_access.intersection(block_dict[day]))
        for block in block_t_access:
            accessTime(block, day)
            absAccessTime(block, day)
    if (day%50==0):
        print("Day", day)
print("Done")

# Removes all of the blocks that did not repeat
removeTimerRedundancies(block_access_timer)
removeTimerRedundancies(block_abs_access_timer)


# In[16]:


# block_diff_access_timer
# keys are unique blocks
# values are the difference between subsequent days of access
block_diff_access_timer = {}
for block in set(block_abs_access_timer.keys()):
    block_diff_access_timer[block] = []

# Populates block_diff_access_timer 
# Iterates through each block and takes the difference between the
# subsequent times of access
for block in list(block_abs_access_timer.keys()):
    block_diff_access_timer[block] = [block_abs_access_timer[block][i + 1]
                                      -block_abs_access_timer[block][i] 
                                      for i in range(len(block_abs_access_timer[block])-1)]


# In[54]:


# Counts the number of blocks that have been accessed consecutively under 
# the given number of days
cons_occurrence_dict = {}
for block in list(block_diff_access_timer.keys()):
    if len(list(block_diff_access_timer[block])) > 0:
        threshold = max(block_diff_access_timer[block])
        if threshold in cons_occurrence_dict:
            for dayCount in range(0, threshold + 1):
                cons_occurrence_dict[dayCount] += 1
        else:
            for dayCount in range(0, threshold + 1):
                cons_occurrence_dict[dayCount] = 1

cons_exact_occurrence_dict = {}
for block in list(block_diff_access_timer.keys()):
    if len(list(block_diff_access_timer[block])) > 0:
        threshold = max(block_diff_access_timer[block])
        if threshold in cons_exact_occurrence_dict:
            cons_exact_occurrence_dict[threshold] += 1
        else:
            cons_exact_occurrence_dict[threshold] = 1


# In[59]:


fig, ax = plt.subplots(1,1)
i = np.arange(1,1000)
exampleList = list(block_access_timer.keys())
for block in list(block_access_timer.keys()):
    i = np.arange(0,len(block_access_timer[block]))
    ax.plot(i, block_access_timer[block], label=("Block: ", block))
ax.legend(title='Block')
ax.set_title('Days Since First Access Vs. Access')
ax.set_ylabel('Days Since First Access')
ax.set_xlabel('Access #')
ax.set_ylim(0, None)
ax.set_xlim(0, None)


# In[61]:


fig, ax = plt.subplots(1,1)
ax.plot(sorted(cons_occurrence_dict.keys()), 
        list(cons_occurrence_dict.values()))
ax.set_title('Block Quantity Accessed Continuously')
ax.set_ylabel('Number of Blocks Accessed Continuously')
ax.set_xlabel('Day Threshold for Continuous Access')
ax.set_ylim(0, None)
ax.set_xlim(0, 60)
plt.savefig("Figure3.png")


# In[ ]:


fig, ax = plt.subplots(1,1)
ax.plot(sorted(cons_exact_occurrence_dict.keys()), 
        list(cons_exact_occurrence_dict.values()))
ax.set_title('Block Quantity Accessed Continuously')
ax.set_ylabel('Number of Blocks Accessed Continuously')
ax.set_xlabel('Day Threshold for Continuous Access')
ax.set_ylim(0, None)
ax.set_xlim(0, 60)
plt.savefig("Figure4.png")


# In[ ]:




