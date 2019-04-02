#!/usr/bin/env python
# coding: utf-8

# In[7]:


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

# In[8]:


# Data collected from a spark query at CERN, in pandas pickle format
# CRAB jobs only have data after Oct. 2017
ws = pd.read_pickle("data/working_set_day.pkl.gz")
# spark returns lists, we want to use sets
ws['working_set_blocks'] = ws.apply(lambda x: set(x.working_set_blocks), 'columns')
ws['working_set'] = ws.apply(lambda x: set(x.working_set), 'columns')


# In[9]:


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


# In[10]:


# join the data tier definitions
datatiers = pd.read_csv('data/dbs_datatiers.csv').set_index('id')
ws['data_tier'] = datatiers.loc[ws.d_data_tier_id].data_tier.values


# In[11]:


date_index = np.arange(np.min(ws.day.values//86400), np.max(ws.day.values//86400)+1)
date_index_ts = np.array(list(datetime.date.fromtimestamp(day*86400) for day in date_index))


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


recall_size['3m'].mean()/1e12


# In[ ]:


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

# Creating a list to keep track of the number of times a setBlock
# (a unique instance of a block) appears in a day
block_occurrence = dict()

print("Initializing block_occurrence")
# This initializes the block_occurrence dictionary with 0 values at each setBlock
for setBlock in block_set:
    block_occurrence[setBlock] = 0
    
for i, setBlock in enumerate(block_set):
    for day in block_dict:
        block_occurrence[setBlock] += block_set.intersection(block_dict[day])
#    print("Day ", day)
    print("Block ", i," out of ",len(block_set))


# In[ ]:


active_days_list = []
for key in blocks_day_lite:
    active_days_list.append(date_index_ts[key])


# In[ ]:


fig, ax = plt.subplots(1,1)
ax.plot(block_occurrence.keys(), block_occurrence.values(), label="Occurrences")
ax.legend(title='Block Occurrences')
ax.set_title('Total Block Accesses')
ax.set_ylabel('Block Occurrences')
ax.set_xlabel('Block Number')
ax.set_ylim(0, None)
ax.set_xlim(0, None)

plt.savefig("Figure3.png")


# In[ ]:

"""
fig, ax = plt.subplots(1,1)
ax.plot(active_days_list, block_occurrence, label="Occurrences")
ax.legend(title='Block Occurrences')
ax.set_title('Block Accesses Per Day')
ax.set_ylabel('Block Occurrences')
ax.set_xlabel('Block Number')
ax.set_ylim(0, None)
ax.set_xlim(0, None)
"""
