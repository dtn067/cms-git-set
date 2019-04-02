Notes
-----
- date_index is a list of indices (that increment by 1 from 17166 to 17926)
- date_index_ts is a list of datetime.date() calls for dates from 2016/12/30 to
  2019/1/29. This is mainly used to show time as the independent axis on plots.
- ws is a pandas.core.frame.DataFrame
- ws.day outputs a panda.core.series.Series
- ws.day.values outputs an ndarray of the raw values
- {k: np.zeroes_like(date_index) for k in lifetimes} defines a dictionary
  where k are the keys and the array of zeroes returned by 
  np.zeroes_like(date_index)
- ws_size, nrecalls, and recall_size are populated in the loop in lines 103-113
- ws_filtered[today] outputs a DataFrame of the data of the specified day
- ws_filtered[today].working_set.blocks is a series of blocks from the working
  set
- blocks_day is a list that is populated with sets of working_set_blocks
- lines 84-87 goes through the date_index list and appends the 
  working_set_blocks corresponding to each date to the blocks_day list
- current is a set of blocks
- recall_size[key][i] gives the block size 

Questions
---------
- What is "spark"? Line 34
- What are datatiers?
- Why are these variables defined with the same code? Line 98-100
- What is recall size?
- What is blocksize? Line 53
