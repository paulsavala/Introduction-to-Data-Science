#!/usr/bin/env python
# coding: utf-8

# # Working with large datasets
# 
# Prior to this course, if you had never done any data analysis you probably had never worked with more than a few hundred rows of data in Excel. Even if you did have experience with data analysis, it's likely that you had never worked with more than tens or hundreds of thousands of rows of data. However, modern companies often work with millions or billions of rows of data. Simply taking the things we've done up to this point in this course and applying them to large datasets will not work. This week we'll see why, and learn some ways to understand and work with large datasets.

# ## Imports

# In[ ]:


import pandas as pd
import numpy as np

import time

from xgboost import XGBRegressor


# ## Understanding memory usage <a id="understanding_memory"></a>
# 
# Anything you run on a computer is limited by two things: processing power (how quickly your computer can perform calculations) and memory (how much data your computer can hold). Each of these two components also have two places where they happen:
# 
# - Processing
#     - **CPU:** (Central Processing Unit) The main processor your computer uses to run pretty much all calculations
#     - **GPU:** (Graphic Processing Unit) A separate component which some (but not all) computers have. These are optimized to perform matrix multiplication. 
# - Memory
#     - **HDD:** (Hard Disk Drive) The main place your computer stores your programs and files. When your phone tells you "you are running out of memory", they are referring to your HDD, or "hard drive" for short.
#     - **RAM:** (Random Access Memory) A much faster form of storage than your HDD. This is used to store data which will only be needed for a short period of time. For example, if you take a picture on your phone it stores it on the hard drive. However, if you want to apply some filters and post it on Instagram, your computer will copy it to RAM. While in RAM, it is much faster to manipulate it (e.g. apply filters to, crop, etc.) than if it were working with the copy on your hard drive.
#     
# In this course we won't deal with GPUs. However, modern deep learning involving neural networks rely heavily on GPUs, since neural networks primarily involve matrix operations. The machine learning we've done in this class wouldn't be faster on a GPU, because we aren't working with matrices. Therefore, we won't worry about the processing component. Memory, on the other hand, is a major factor with every machine learning project, no matter if it's using decision trees, neural networks, or just linear regression.
# 
# Thus far, the datasets we've worked with are fairly small. But before we get ahead of ourselves, how do we even measure "small" vs "big"? Your first guess is probably the number of rows. However, rows and columns are the same thing, so one million rows and ten columns is the same as ten rows and one million columns. However, even if two datasets have the same number of rows and columns, they can still be drastically different sizes. In fact, if two people are given the *exact same dataset*, depending on how they load them into Pandas one can easily be *ten times* the size of another! Let's load the free throws NBA data to see this in action.

# In[ ]:


df = pd.read_csv('data/free_throws.csv')


# In[ ]:


df.head()


# In[ ]:


print(f'Rows = {df.shape[0]}\nColumns = {df.shape[1]}\nTotal cells = {df.shape[0] * df.shape[1]}')


# In order to see how much memory some data is using, you can use the Pandas method `.memory_usage()`. This will return the memory usage in bytes (explained below).

# In[ ]:


df.memory_usage()


# As you've likely at least heard, computers store all information in binary (except quantum computers!). That is, everything is stored as a `0` or `1`. A value of `0` or `1` is referred to as a **bit**. A **byte** (pronounced "bite") is eight bits. So, since a single bit has 2 possible outcomes, a byte has $2^8 = 256$ possible values. Historically every single number, character or punctuation mark on a computer could be stored with one byte ([here](https://www.commfront.com/pages/ascii-chart) is a chart showing how every possible value was stored). Therefore, a byte is considered to be a fundamental unit of memory measurement. While bits are smaller and make up bytes, people today primarily work directly with bytes (in the same way as scientists typically work with molecules, cells, etc., as opposed to directly counting the number of protons, neutrons and electrons). Bytes are very small, and most modern data/programs is instead measured in larger measurements. The next largest measurement is a "kilobyte". A **kilobyte** is one thousand bytes. Next we have a "megabyte". Historically, a **megabyte** was $2^{20} = 1,048,576$ bytes. However, in modern times we simply call a megabyte one million bytes (hence the prefix "mega", which means "million"), or $10^6$. Thus a megabyte can store one million characters (ornumbers, or punctations, etc.). Next up is gigabytes. A **gigabyte** historically was $2^{30} = 1,073,741,824$ bytes, however we now round that to one billion bytes, or $10^9$. Measurements certainly go higher, but we'll stop there for now.
# 
# As mentioned above, Pandas `.memory_usage()` displays the value in bytes. This is too small of a measurement. Let's convert it to megabytes by dividing by $10^6$.

# In[ ]:


df.memory_usage() / 10**6


# So each column (besides the index) is using about 5 megabytes (or MB, for short). To find out the *total* memory usage, just sum these up.

# In[ ]:


(df.memory_usage() / 10**6).sum()


# So about 64 MB in total. Let's look at the data types of each column.

# In[ ]:


df.dtypes


# We see that there are a mixture of integers, strings (objects) and floats. One component we never discussed was the "64" after `int` and `float`. This means that the computer can store $2^{64}$ values of that type. Since we potentially have both positive and negative values, this means that `int64` can store values from $-2^{32}$ to $2^{32}$, or values in the range $(-4294967296, 4294967295)$. Why does that matter? Because Pandas doesn't know if you any number you put in will actually *need* to be that big or not. Therefore, it *allocates* (sets aside) that much space for every cell in your data. Is that necessary? After all, the `shot_made` column only has a value of 0 or 1!
# 
# A very powerful technique is to **downcast** your data by storing it in the smallest possible data type you can. In the background, Pandas is actually using numpy to store all data, so we're really interested in the numpy data types. [Here is a list](https://numpy.org/devdocs/user/basics.types.html) of all numpy data types. Take a moment and find the integers on that chart. You will see that there are several different data types to store integers. We will focus on the following ones:
# - np.int8: (-128 to 127)
# - np.int16: (-32768 to 32767)
# - np.int32 (-2147483648 to 2147483647)
# - np.int64: (-9223372036854775808 to 9223372036854775807)
# - np.uint8: (0 to 255)
# - np.uint16: (0 to 65535)
# - np.uint32: (0 to 4294967295)
# - np.uint64: (0 to 18446744073709551615)
# 
# The `u` in front of some of these refers to "unsigned" which means it is positive (because you don't need a "sign" to represent it). As you might guess, data types for smaller numbers take up less space in memory. Let's start with the `period` column, which is currently stored as `int64` (see the results of `df.dtypes` above). Let's check what values actually appear in that column:

# In[ ]:


df['period'].unique()


# Only 1 through 8. Therefore, it's positive and small, and so we can store it in the smallest unsigned integer possible, namely `np.uint8`. In order to change the data type, you can use the Pandas method `.astype()`.

# In[ ]:


df['period'] = df['period'].astype(np.uint8)


# If you look at the data, you'll see that nothing has changed:

# In[ ]:


df.head()


# All that's happening is that the computer is setting aside less memory to store this value. That's because now it nows that (at worst) it will need to store a value up to 256, as opposed to 922,337,203,685,477,5807. Let's now look at the memory usage again.

# In[ ]:


df.memory_usage() / 10**6


# We see that `period` went from 4.94 MB to 0.62MB! That's a reduction of 4.94 / 0.62 = 7.9 times! Let's do the same for the other integer columns. The only integer column we need to be careful about is the `season_start`. Since the seasons are in the 2000's, just representing numbers up to 256 won't be enough. In fact, if you try to downcast it to `np.int8` you get the following:

# In[ ]:


df['season_start'].astype(np.int8)


# So we'll put that one as `np.uint16` so that values up to 65,535 can be stored.

# In[ ]:


small_int_cols = ['playoffs', 'shot_made', 'home_score', 'visit_score', 'home_final_score', 'visit_final_score', 'shot_count']
big_int_cols = ['season_start']

for col in small_int_cols:
    df[col] = df[col].astype(np.uint8)
    
for col in big_int_cols:
    df[col] = df[col].astype(np.uint16)


# Checking memory usage again:

# In[ ]:


df.memory_usage() / 10**6


# In[ ]:


(df.memory_usage() / 10**6).sum()


# Not bad! We went from about 64 MB to 26 MB, a reduction of more than half! Float and object columns (along with dates) are typically left alone. The one thing you can do is turn strings into numbers (such as by using a label encoder), which will reduce their memory usage. However, in our case that's not worth the effort. One final note is that a way to store data which is only `1`/`0` is a boolean (`True`/`False`). Let's make `shot_made` be a boolean.

# In[ ]:


df['shot_made'] = df['shot_made'].astype(np.bool_)


# In[ ]:


df.memory_usage() / 10**6


# As you can see, this didn't actually reduce the memory usage at all. All it did was change how it's displayed:

# In[ ]:


df.head()


# ## Selectively loading data <a id="selective_loading"></a>
# 
# If you go look at the [competitions on Kaggle](https://www.kaggle.com/competitions) you will see many that have datasets in the tens to hundreds of gigabytes. You may think this is not a problem, as modern computers also have hundreds of gigabytes of storage. In fact, if you were to go buy the [cheapest computer Apple makes](https://www.apple.com/shop/buy-mac/mac-mini) it comes with at least 256 GB of storage, and can easily be scaled to several factors of that. However, not all storage is the same. As discussed above, there is both your hard drive (HDD) and RAM. The Mac Mini linked above has 256 GB hard drive (in fact an SSD drive, which is a faster version of a hard drive) and 8 GB of RAM. 
# 
# When you download data (or any file) it is stored on your hard drive. However, once you start working with it in Pandas, it copies it onto your RAM. This is because RAM is much faster. Therefore, if it needs to quickly calculate things from your data, find a subset, build a model, etc., this is all much faster if it is on RAM. So while your computer may have 256 GB of storage and can easily hold a 100 GB dataset, the place where your computer *wants* to put your data (RAM) is only 8 GB.
# 
# Because of this, working with large datasets is extremely difficult. We saw above that one way to deal with this is to load the entire dataset and then change the data types to make it smaller. However, what if the entire data is too large to load? Or what if it's small enough to load, but still would take up lots of memory on your computer and make everything slow? There are two ways to deal with this.

# ### Loading only a subset of your data <a id="loading_subset"></a>
# 
# One solution is just to load a subset of the data. For example, in the NBA data above, is it really necessary for us to have access to *every single free throw* in order to predict free throws? In fact, when we build models we typically only take 70% or so to train the model on. If you want to only load a subset of your data, you can do so using the `nrows` parameter in `.read_csv()`. So for example, the code below loads only the first 100 rows of the data.

# In[ ]:


df_mini = pd.read_csv(drive_dir + 'data/free_throws.csv', nrows=100)


# In[ ]:


df_mini.head()


# In[ ]:


df_mini.shape


# This can be useful for just working with a small subset of your data, or for getting just a few rows to look at. After all, how can you you know what data types to make each column if you can't even see the data?
# 
# You may be tempted to just say "if my data is big, I'll just load 100,000 rows and use that for my training set". The problem is that `nrows=100000` simply takes the *first* 100,000 rows. So in our NBA example, that would start with the first season. But what if things changed in later seasons? That data wouldn't be in our training data, and thus our sample wouldn't be very representative of the data as a whole. It is possible to tell Pandas to skip every (say) 10th row, but even then you run the risk of getting a poor sample. Therefore, it is to your advantage to simply load as much data as you can and understand your data first. Then you can decide how best to sample your data.
# 
# Finally, suppose you looked at the first 100,000 rows and realized that the dates are increasing. Now you want to get the next 100,000 rows so that you can get some data from later dates. How do you do this? Do you have to load 200,000 rows and just look at the last 100,000? If so, what do you do when you need rows 10,100,000 to 10,200,000? Luckily, Pandas let you `skip_rows`.

# In[ ]:


# Load 100 rows again, but skip the first 100 so that we're starting at row 101
df_mini2 = pd.read_csv(drive_dir + 'data/free_throws.csv', nrows=100, skiprows=100)


# In[ ]:


df_mini.tail()


# In[ ]:


df_mini2.head()


# You may have noted by when we skipped rows it also skipped the headers (rows showing the column names at the top), so that our columns have names like "2006" and "BOS". The easiest thing to do is to tell Pandas to skip from rows 1 to 100 (say).

# In[ ]:


df_mini2 = pd.read_csv(drive_dir + 'data/free_throws.csv', nrows=100, skiprows=(1, 100))


# In[ ]:


df_mini2.head()


# With this, we can now do something called "chunking". **Chunking** refers to loading the data in consecutive chunks, and doing something with each chunk. For example, we could load the rows 100 at a time, and sample 10 of them.

# In[ ]:


# Sample from the first 100 rows
sample_df = pd.read_csv(drive_dir + 'data/free_throws.csv', nrows=100).sample(10)

# Load 100 rows 20 times
for i in range(20):
    new_sample_df = pd.read_csv(drive_dir + 'data/free_throws.csv', nrows=100, skiprows=(1, (i+1)*100)).sample(10)
    sample_df = pd.concat([sample_df, new_sample_df])


# In[ ]:


sample_df.head()


# In[ ]:


sample_df.shape


# At the end of this notebook we'll discuss a way to make this more efficient.

# ### Setting data types upon loading <a id="loading_dtypes"></a>
# 
# Now that you know how to load just a few rows at a time, you can examine your data to see what data types would work best to save as much memory as possible. So for instance, we see that all integer columns except scores could be loaded as `np.uint8`, and the scores as `np.uint16`. We can specify this upon loading by giving a dictionary with column names and data types to the `dtype` argument in `.read_csv()`. Note that you don't have to specify *every* columns data type. If you leave a column out, Pandas will just load it with what it thinks is best. So since it's already correctly loading the player names, team names and minutes, we'll leave those alone. We'll just tell is to load the integers using smaller memory data types.

# In[ ]:


dtype_dict = {'period': np.uint8, 'playoffs': np.uint8, 'shot_made': np.uint8, 'home_score': np.uint16, 'visit_score': np.uint16, 'home_final_score': np.uint16, 
             'visit_final_score': np.uint16, 'season_start': np.uint16, 'shot_count': np.uint8}

df_slim = pd.read_csv(drive_dir + 'data/free_throws.csv', dtype=dtype_dict)


# In[ ]:


df_size_mb = (df.memory_usage() / 10**6).sum()
df_slim_size_mb = (df_slim.memory_usage() / 10**6).sum()

print(f'Original file memory usage: {df_size_mb:.1f} MB')
print(f'Slimmed file memory usage: {df_slim_size_mb:.1f} MB')
print(f'Reduction of {df_size_mb / df_slim_size_mb:.1f}x')


# ## Timing computations <a id="timing_computations"></a>
# 
# Up to this point we've focused on memory usage. Now, let's turn our attention to processing time. When you first started this course you were probably struggling to just learn how to get your code to do what you want. But now that you know more you are likely able to accomplish lots of different tasks, often times in more than one way. Once you reach that point it's good to start thinking about computational efficiency. Simple things like linear regression happen almost instantaneously, but more complex tasks like grid search cross validation using XGBoost can easily take hours, if not days to run. In addition, often you will run a long computation, only to realize that you should have done some things differently. Perhaps you picked a range of hyperparametrs to search, ran grid search for four hours, and the results came back that all the hyperparameters selected were the largest ones you let it search. Maybe that means you need to go larger! So you start grid search again with even larger hyperparameters, which means it will be running for even longer! In cases like these it is beneficial to try and speed up your code.
# 
# The first question to ask then is, how long does my code take to run? Python can help us with that! The simplest way, which is appropriate for anything which will take several seconds or longer (as opposed to milliseconds or nanoseconds, for which it won't be as precise) is using the built-in Python library `time`, which we imported up top.

# In[ ]:


# This is _not_ the most efficient way to check for primality, but that's okay for our purposes
def is_prime(n):
    for c in range(2, n):
        # The percent operator "%" means "modulo", so it's dividing by c and taking the remainder
        if n % c == 0:
            return False
    # If nothing divides it, return True
    return True


# In[ ]:


is_prime(4)


# In[ ]:


is_prime(5)


# In[ ]:


# Record when the computations start
start = time.process_time()

# Do your computations
primes = []
for n in range(2, 10**5):
    if is_prime(n):
        primes.append(n)

print(f'{len(primes)} primes found')
print(f'{100*len(primes) / (10**5 - 1):.2f}% of the numbers between 2 and 10^5 are prime')

# Record when the computations end
end = time.process_time()

# Print out how long it took (the result is in seconds)
print(f'Elapsed time: {end - start:.3f} seconds')


# ## Comprehensions <a id="comprehensions"></a>
# 
# In the very first notebook of this semester we introduced list comprehensions. Recall that list comprehensions are lists (you can also do them with sets, dictionaries, and any iterable) that you create using the format `[f(a) for a in blah]`. Why do we do this? Is it just to be fancy? Let's time some operations to see how creating a list of primes this way compares to doing it using a `for` loop.

# In[ ]:


# Method 1 - for loop
start = time.process_time()

squares = []
for n in range(2, 10**8):
    squares.append(n**2)
        
end = time.process_time()

print(f'Elapsed time using a for loop: {end - start:.3f} seconds')


# In[ ]:


# Method 2 - list comprehension
start = time.process_time()

squares = [n**2 for n in range(2, 10**8)]

end = time.process_time()

print(f'Elapsed time using a for loop: {end - start:.3f} seconds')


# That's about an 8% reduction (at least on my computer) in run time. While that may not seem like a lot, 8% of six hours is about half an hour saved. If you have multiple `for` loops in your code (or even worse, a `for` loop inside another `for` loop, called a "nested `for` loop") then you are losing a lot of computation time.

# ## Deleting data in-memory
# 
# Whenever you load data into your computer using `pd.read_csv()` it stays in memory. For example, Python is still holding the free throw data above in memory waiting for us to use it. However, sometimes you may load some data, look at it, keep a subset of it, and then want to throw away the rest. How do you tell Python that you no longer want it around? The answer is `del` (short for delete). Deleting data is as simple as `del my_data`. Let's check it out.

# In[ ]:


# Check to make sure the data is still in memory
df.head()


# In[ ]:


# Delete it
del df


# In[ ]:


# It's no longer in memory
df.head()


# ## Sklearn warm start <a id="warm_start"></a>
# 
# Let's suppose you had a large dataset that you couldn't fit in memory. In that case, you decide to load the data in chunks and create a subset to train your model on. However, after looking at your data you realize that even your *subset* would be too large. What can you do about it? You may guess that you can load data, train your model, load more data, train your model again, and so forth, as such:

# In[ ]:


# Load the first 5 groups of 1000 rows
n_rows = 1000
n_chunks_to_load = 5

# Instantiate the model
xgb_reg = XGBClassifier()

# Load data, train the model, repeat
for i in range(n_chunks_to_load):
    print(f'Fitting model on rows {n_rows*i} to {n_rows*(i+1)}')
    sample_df = pd.read_csv(drive_dir + 'data/free_throws.csv', nrows=n_rows, skiprows=(1, (i+1)*n_rows))
    X = sample_df[['shot_count', 'playoffs', 'minutes']]
    y = sample_df['shot_made']
    xgb_reg.fit(X, y)


# This code runs, and you may think everything has gone perfectly. However, what you've *actually* done is train a model *only on rows 4000 to 5000*. This is because every time you call `.fit()`, sklearn assumes you want to start fresh. So even if you already `.fit` your model on some other data, it assumes you no longer want that, and so it creates a new one and throws away the old one. What we want is to do a *warm start*. In model training, a **warm start** is when you take a model which was already trained on some data, and then train it on additional data *without discarding what it already learned*. It is incredibly simple to do. When you instantiate the model, simply set `warm_start=True`. We can copy and paste the above code with that one minor change, and it will work how we want it to.

# In[ ]:


# Load the first 5 groups of 1000 rows
n_rows = 1000
n_chunks_to_load = 5

# Instantiate the model
xgb_reg = XGBClassifier(warm_start=True) # THIS IS THE ONLY CHANGE!

# Load data, train the model, repeat
for i in range(n_chunks_to_load):
    print(f'Fitting model on rows {n_rows*i} to {n_rows*(i+1)}')
    sample_df = pd.read_csv(drive_dir + 'data/free_throws.csv', nrows=n_rows, skiprows=(1, (i+1)*n_rows))
    X = sample_df[['shot_count', 'playoffs', 'minutes']]
    y = sample_df['shot_made']
    xgb_reg.fit(X, y)


# ## Final thoughts <a id="final_thoughts"></a>
# 
# Most large companies who hire data scientists or analysts have incredible amounts of data to work with. Early on as a data scientist I was told that I would be working with a "small" dataset that had "only" eighty million rows of data. When working with data of this size you have to think strategically. Simply trying to load the entire data on your computer at once will never work. Instead, start with a small subset of the data to get a feel for it. Then, start figuring out what subset of the data you want to work with. Is it fine to just load the first 100,000 rows? Or do you need to sample according to a date which is growing? Come up with your strategy, and then import a chunk of the data. From that data, sample what you want and set it aside. Then, delete the original data using `del`. Then, load the next chunk and repeat. Note that this is different from what we did earlier, in that we're deleting the data we loaded originally. This is important, because otherwise we are just accumulating more and more data in memory, and we're actually not being any more efficient than just loading the entire thing!

# In[ ]:




