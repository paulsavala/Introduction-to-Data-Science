#!/usr/bin/env python
# coding: utf-8

# # First steps with Pandas and Jupyter notebooks

# ## Jupyter notebooks
# 
# What you're looking at right now is what's called a Jupyter notebook. When you took your programming course, you probably wrote your code in a text editor and then ran it from the command line. Modern data scientists typically start their work in Jupyter notebooks. A notebook is a collection of code cells (like the one at the top of this notebook) which run Python code, and Markdown cells (like this one!) which have text.
# 
# If you've ever used Mathematica or similar math systems, this is a very similar setup. Rather than writing your code in the text editor, leaving to run it in a console, and then coming back to change it, everything happens in one place.

# ## Jupyter notebook cheat-sheet <a id="jupyter"></a>
# To edit a cell, click on it (if it's a `Code` cell), or double-click on it (if it's a `Markdown` cell). You can change a cell between `Code` and `Markdown` by clicking on the cell, then using the dropdown at the top of this worksheet. You can also click to the left of the cell and hitting `C` on your keyboard for `Code`, or `M` for `Markdown`. To run a cell, select the cell, then hit `Shift-Enter`. The output will appear below it. Not all commands will generate an output, so your output can be blank.
# 
# Cells are just containers for tiny Python programs. So you can do something like `3+2` and run it with `Shift-Enter` and get an output of `5`. You can also write `if` statements, define new functions, classes, etc. Anything that works in Python works here too.
# 
# ```{note}
# You'll need to either download this notebook or run it in Binder in order to execute cells!
# ```

# In[1]:


# This is a code cell. Click in it to edit it.
# Try running some small Python code here (like adding two numbers).
# Then, click Shift-Enter to run it. You should see the output below this cell.


# ## Markdown cheat-sheet
# 
# Markdown is a way to type pretty text using bold, italics, links, codeblocks and more, without needing to use a word processor like Microsoft Word. There is a little bit of memorization, but it's not too bad. Below are examples of the most common things you'll use. For a more detailed list, checkout [this cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet).
# 
# ### Double-click on this cell to see a breakdown of how I accomplished all these things.
# 
# *This is italic*
# 
# **This is bold**
# 
# # This is a big header
# 
# ## This is a slightly smaller header
# 
# ### You can pretty much go as small as you want
# 
# - You can also make a list by using dashes
# - You can make as many as you want
# 
# 
# 1. If you want numbers instead
# 2. Just use numbers
# 
# You can also makes links as follows: [The title for your link](www.stedwards.edu)
# 
# If you want to type something that uses Python, and you want to remember that this is a Python *command* (as opposed to just some text), it's good practice to surround it with backticks. Backticks are the thing in the top-left of your keyboard, on the same key as the tilde ~. 
# 
# As an example, suppose you were writing something up showing someone how to use the print command in Python. You could write the following: 
# 
# "In order to add something in Python, just type the numbers. For example, you could teach someone how to print in Python by telling them to type `print(3+2)`."
# 
# See how surrounding it with backticks made it look different? It makes it clear that this is a Python command, and not just some random text.

# ## Jupyter notebook best practices
# - Always title your notebook.
# - Break your code into sections. Start by doing your imports, then your next section might be loading the data, and so forth. Title each section using Markdown, and give each section a description of what you're doing there.
# - Keep your code clean. It's okay to create new cells to test things, but once your testing is done and those cells are no longer needed, delete them. Go back to old working cells and clean them up. Write the code nicer, put a Markdown cell describing what's going on, etc.
# - Leave yourself notes about what you've done. *Right now* you remember what you've done, but you won't remember it a month from now.
# - If you find yourself copying and pasting some code over and over, make it into a function and call that function instead.

# # Python <a id="python"></a>
# The following is a brief review of Python. Use it to refresh your memory and to help you with the homework.

# ### Data types <a id="data_types"></a>
# In Python, the most common ways to represent data (called **data types**) are as one of the following:
# - `int`: An integer (whole number, positive or negative)
# - `float`: A decimal
# - `str`: A string (characters, such as "abc")
# 
# In order to see what data type something is, you can just type `type(my_thing)`.

# In[2]:


type('abc')


# In[3]:


type(123)


# In[4]:


type(123.45)


# ### Data structures <a id="data_structures"></a>
# When you want to store data, there are several ways to do it. The simplest way is just to store it in a variable. You can do this by typing `my_variable = some_value`. In a Jupyter Notebook (the thing you're working in right now), you don't necessarily need to type `print` if you want to look at a variable. You can just type the variable name and run the cell.

# In[5]:


x = 3
x


# Since I put the `x` as the last thing in the cell, Jupyter printed it out for me to see what it is.

# Often you will need to store several variables/pieces of data in an organized way. The most common ways to do this are the following **data structures**:
# - `list`: A list of values. Lists are created using brackets (`[` and `]`).
# - `set`: Similar to a list, but there is no order, and each object shows up at most once. Created using curly braces (`{` and `}`).
# - `dict`: A dictionary. This stores things with a `key` and a `value`. Dictionaries are created using curly braces.

# In[6]:


my_list = [1, 1, 2, 3, 'a', 'b', 'c', 'c']
# The set and list look the same at first, but once you print them you will see the difference.
my_set = {1, 1, 2, 3, 'a', 'b', 'c', 'c'} 
my_dict = {'last_name': 'Savala', 'age': 36, 'favorite_number': 3.14}


# In[7]:


my_list


# In[8]:


my_set


# In[9]:


my_dict


# In[10]:


my_dict['age']


# ### Control structures <a id="control_structures"></a>
# Often you want to control how some events happen. This could mean things like `if` today is Saturday then don't do homework, or something like `for` each number in this list, check if it's prime. These are called **control structures**. The most common control structures in Python are `if`/`elif`/`else` statements, and `for` loops (there are more!).

# `if`/`elif`/`else` structures work just like you think. Remember to put a colon `:` after each one, and indent everything that happens in each part.

# In[11]:


x = 5
if x < 3:
    print('Somehow, 5 is less than 3')
elif x > 7:
    print('How is it that 5 is greater than 7?')
    print('That makes no sense!')
else:
    print('Whew, guess everything is okay')


# `for` loops take some structure (often a `list`) and iterate through it. This is useful when you want to work with every element in a list.

# In[12]:


my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for x in my_list:
    print(x**2)


# ## Functions <a id="functions"></a>

# ### Defining functions <a id="defining_functions"></a>
# Writing code as functions is incredibly important in any programming language. This should seem familiar to math people, as we deal with functions all the time! In Python, write a function using the `def` command. You then put the name of the function, and in parentheses all the inputs the function takes. Don't forget to include a colon `:` at the end, and to indent everything inside the function. At the end you can (but don't have to) use the `return` statement to return some value.

# In[13]:


# A function with no return statement
def add_and_print(x, y):
    print(x + y)


# In[14]:


add_and_print(3, 4)


# In[15]:


# A function with a return statement
def add_and_return(x, y):
    return x + y


# In[16]:


add_and_return(3, 4)


# These two look the same, but there's an important difference. If we `return` a value, then we can store that value and use it later. If we simply `print` the value, that's not possible. Example:
# 

# In[17]:


z = add_and_return(3, 4)
print(z)
print(2 * z)
print('Now I can manipulate z however I want')


# ```{warning}
# An error is coming!
# ```

# In[19]:


# We will get an error because add_and_print doesn't return anything, 
# so the program has no idea what value to save to z.
z = add_and_print(3, 4)
z = 2 * z
z


# ### Calling functions <a id="calling_functions"></a>
# 
# Functions have **arguments**, which are simply the inputs. When you define the function all the arguments must have names. That's why we used `x` and `y` when we defined `add_and_print` above. Here's another example:

# In[21]:


def my_name(first_name, middle_name, last_name):
    print('Your first name is ' + first_name)
    print('Your middle name is ' + middle_name)
    print('Your last name is ' + last_name)


# When you call this function, you can choose to use those argument names or not. That is, either of the following two are exactly the same:

# In[22]:


my_name(first_name='Paul', middle_name='Awesome', last_name='Savala')


# In[23]:


my_name('Paul', 'Awesome', 'Savala')


# Note that in the first one we directly wrote `first_name="Paul", ...`, while in the second we simply typed them in. **If you do not use the argument names, you type the arguments in the order the function definition has them.** Since the function has `first_name` as the first argument, then it just assumes it's the first argument (`first_name`), that the second is the second argument (`middle_name`), and so forth. **If you do supply the arugment names then the order doesn't matter.** Here are some examples:

# In[24]:


# This is wrong!
my_name('Savala', 'Awesome', 'Paul')


# In[25]:


# This works though
my_name(last_name='Savala', middle_name='Awesome', first_name='Paul')


# Either way is just fine, it's totally up to you. If your function has many arguments then it's good practice to include the argument names when you call it. That way the person reading your work isn't trying to remember/guess what the seventh argument in a big list of arguments is. But if it's just one or two then omitting the argument name is probably just fine.

# ### Default values <a id="default_values"></a>
# 
# In a function you can supply **default values**. For instance, not everyone has a middle name. So in the function above we would like middle name to be optional. We can do this by setting a default value of, say, `None`. To do this, simply put the default value there when you define the function:
# 
# ```{warning}
# Here comes an error!
# ```

# In[29]:


def my_name(first_name, middle_name=None, last_name):
    print('Your first name is ' + first_name)
    if middle_name is not None:
        print('Your middle name is ' + middle_name)
    print('Your last name is ' + last_name)


# As you can see from the error, arguments *without* a default value (like `first_name` and `last_name` in our example) need to come first, then arguments *with* default values must come last. So let's fix that.

# In[30]:


def my_name(first_name, last_name, middle_name=None):
    print('Your first name is ' + first_name)
    if middle_name is not None:
        print('Your middle name is ' + middle_name)
    print('Your last name is ' + last_name)


# In[31]:


my_name(first_name='Paul', last_name='Savala')


# ### f-strings <a id="f_strings"></a>
# If you want to print text, but also include a variable, there are three ways to do it.

# In[32]:


# Method 1 (the old "format" way)
my_age = 36
my_name = 'Paul'

'My name is {name} and I am {age} years old'.format(name=my_name, age=my_age)


# In[33]:


# Method 2 (use +)
'My name is ' + my_name + ' and I am ' + str(my_age) + ' years old'


# Both of these methods works, but they are ugly and slow. In addition, for "method 2" to work, everything has to be a string. So if you removed `str(...)` from `my_age` it would give an error. 
# 
# A much cleaner way which was introduced with Python version 3.6 is called f-strings (the "f" stands for "format"). All you need to do is put an `f` in front of your string. Then, in the curly braces write the name of your variable. Here's an example:

# In[34]:


# Note the "f" in front of the string. That tells Python this is an f-string.
f'My name is {my_name} and I am {my_age} years old'


# f-strings are very useful for printing all kinds of variables, including numbers. However, floats with many numbers after the decimal can print out annoyingly long. For instance, consider the following example:

# In[35]:


f'If a pizza costs $22 and is divided between 7 people, then each person pays ${22/7}'


# Do we really need to see how much each person pays down to the billionth of a penny? We would like just two decimal places. We can do this by including a **format** at the end. The one we will use the most often is `.2f`. The dot "." refers to the decimal. The "2" refers to two places after the decimal. The "f" refers to the number being a float (if you don't put this it will sometimes print the number in scientific form, which is probably not what you wanted). To tell Python you want to use a format, just put a colon ":" after your computation, and then put the format you want. For example:

# In[36]:


# Notice the change in the colons where the calculation occurs
f'If a pizza costs $22 and is divided between 7 people, then each person pays ${22/7:.2f}'


# ### Range <a id="range"></a>
# Suppose I have a list of 5 elements, and I want to know which ones are odd. So for instance, if `my_list = [4, 1, 1, 8, 9]`, I want to know that the second, third and fifth elements are odd. Remember that in Python these **indices** start by counting at zero. So the `indices_with_odd_elements = [1, 2, 4]`. The easiest way to access the incides in a list is to use the `range` function. By default a `range` is it's own thing. In order to be able to better see what is going on, it is useful to convert it to a list.

# In[37]:


range(10)


# In[38]:


list(range(10))


# In[39]:


list(range(1, 10))


# In[40]:


list(range(1, 10, 2))


# In[41]:


my_list = [4, 1, 1, 8, 9]
for i in range(len(my_list)):
    print(f'The index is {i} and the value is {my_list[i]}')


# ### List comprehension <a id="list_comprehension"></a>
# One of the most useful and important ideas in Python is list comprehension. List comprehension allows you to modify a list element-by-element, without needing to use a `for`-loop. Here is an example where we add one to each number in a list. We first show how we would do it with a `for`-loop, and then next show the better way to do it (using list comprehension). We will write functions to create a new list `add_one_list` which is the original list, but with one added to each element.

# In[42]:


my_list = [1, 5, 6, 2, 7, 5, 8, 2]
add_one_list = []
# This setup is slow and ugly
for i in range(len(my_list)):
    add_one_list.append(my_list[i] + 1)

add_one_list


# A better way is to use list comprehension. To do list comprehension, you simply write `[what_I_want_to_do for x in my_list]`. For example:

# In[43]:


my_list = [1, 5, 6, 2, 7, 5, 8, 2]

add_one_list = [x + 1 for x in my_list]

add_one_list


# It's worth noting that Python has "comprehension" for things beyond lists. You can also do it for dictionaries and sets.

# In[44]:


my_list = ['a', 'b', 'c', 1, 2, 3]

int_str_dict = {'str': [x for x in my_list if type(x) == str], 
                'int': [x for x in my_list if type(x) == int]}

int_str_dict


# You should try to use list (or other types of) comprehension as much as possible. They are much faster than loops, and are generally easier to read. This will help greatly when you work with datasets with hundreds of thousands of rows of data.

# ## Using packages <a id="using_packages"></a>
# 
# While Python comes with many functions built in, it doesn't have *everything* could ever want. For example, let's say we wanted to compute sine of an angle. We might try something like `sine(1)`, or `sin(1)` (I'm using radians here). Let's see what happens.
# 
# ```{warning}
# Here comes an error!
# ```

# In[45]:


sine(1)


# In[46]:


sin(1)


# Python doesn't have a "sine" function. We could perhaps write one ourselves. For example, sine has a series definition given by
# $$
# \sin(x) = \displaystyle\sum_{n=0}^{\infty} \frac{(-1)^n}{(2n+1)!}x^{2n+1}
# $$
# However this is a lot of work, and would also require writing a factorial function, and probably eventually a $\pi$ function as well. Luckily for us, other people have already done the hard work! They have done so by writing **packages**. A Python package is simply a collection of code that you can use. Rather than having to copy-paste code you **import** from these packages. For example, there is a `math` package we can use. We will import it as such:

# In[48]:


import math


# As you can see, nothing happened. However, we can access functions inside of this package by using **dot notation**. For example, to use the sine function we would type:

# In[49]:


math.sin(1)


# To access $\pi$ we could do:

# In[50]:


math.pi


# And to use a factorial we could write:

# In[51]:


math.factorial(7)


# Packages are extremely powerful. When you are considering writing some complex code, you should first search to see if someone has already written a package which does it for you.
# 
# When importing from packages, you don't necessary need to import the entire package, which is what we did when we typed `import math`. Some packages are very large, and importing them makes your program slower. Say, for instance, that we only wanted to use the sine function. We could type the following to only import that one function:

# In[52]:


from math import sin


# In[53]:


sin(2)


# If you only plan to use one or two functions from a package, it is good practice to only import those functions. That keeps the code clean and fast.

# ## Python: Homework
# For each of the following questions, write a function which does what is asked.
# 
# 1. Print out the first 20 even numbers
# 2. Test whether or not a number is prime
# 3. Given a list of integers, return a list containing only the odd numbers
# 4. Given a list of integers and a target number, return True if two numbers in that list add up to the target number, or False if nothing does. For instance, `list_of_integers = [1, 4, -4, 2, 8]` and `target=6` we would return `True`, because `4+2=6` and both 4 and 2 are in `list_of_integers`. But we would return `False` if `target=7`, because no two numbers in the list add up to 7.
# 5. Given a string `s` with the first letter capitalized (such as "Math"), return the reversed string (backwards), but fix the capitalization (so it should return "Hatm").
# 6. Use list comprehension (described above) to square every element in a list.
# 7. Use list comprehension to replace each element in the list of positive integers with the word "even" if it is even or "odd" if it is odd. So if the input is `[1, 5, 3, 6]`, it should return `['odd', 'odd', 'odd', 'even']`.
# 8. Given a list of positive integers, return a dictionary showing the even and odd numbers. For example, if the input is `[1, 5, 3, 6]`, then the return should be `{'odd': [1, 5, 3], 'even': [6]}`.

# # Pandas <a id="pandas"></a>

# ## Imports
# Pandas is the library which all data scientists use to manipulate data. We need to tell Python to *use* Pandas. The way to do that is to import it by typing `import pandas`. Then when we want to use it, we type `pandas.thing_I_want_pandas_to_do`, just like we did with the `math` package above. However, typing `pandas` over and over is a bit cumbersome. So it's common to **alias** pandas by giving it a shorter name. The common name is `pd`. So we'll `import pandas as pd`, which means that we can now work with Pandas by typing `pd.thing_I_want_pandas_to_do`, which saves us a bit of typing.

# In[54]:


import pandas as pd


# ## Loading the data <a id="loading_data"></a>
# I have my folders setup so that I have a `data` folder where I keep all my data. So, when I load my data in the next step, I need to tell Pandas to look in the `data` folder. If you *don't* have a data folder and the data is just sitting in the same folder as your current notebook, then you don't need that `data/` part. Dealing with directories can be a bit of a headache, so it's good practice to always create a `data` folder, and put your data in there.
# 
# This data is taken from the [Titanic Kaggle competition](https://www.kaggle.com/c/titanic/data), and describes the survival (or not) of passengers aboard the Titanic. Here's a brief description of the data:
# 
# |Variable|Definition|Key|
# |---|:---:|:---:|
# |survival|Survival|0 = No, 1 = Yes|
# |pclass|Ticket class|1 = 1st, 2 = 2nd, 3 = 3rd|
# |sex|Sex||
# |Age|Age in years||
# |sibsp|# of siblings / spouses aboard the Titanic||
# |parch|# of parents / children aboard the Titanic||
# |ticket|Ticket number||
# |fare|Passenger fare||
# |cabin|Cabin number||
# |embarked|Port of Embarkation|C = Cherbourg, Q = Queenstown, S = Southampton|
# 
# Pandas stores data as things called **DataFrames**, so it's typical to call the variable you load the data into `df`. If you have a bunch of different datasets you're loading, you can tweak the name to help you remember what the data is. So this could be called `titanic_df`, or `train_df`. For now, we'll just use `df`.

# In[55]:


df = pd.read_csv('data/titanic.csv')


# Let's break this command down:
# - `pd` says that we are using a function from Pandas, which we aliased as pd
# - `read_csv` is the function we are using from Pandas. It reads in a CSV (command-separated value) file. If you've never heard of a CSV file, it's basically just an Excel file.
# - `drive_dir` is the Google drive folder. It's created in the very first cell of this notebook (look at the very top).
# - `data/` is the folder where my data is stored in Google drive
# - `titanic.csv` is the name of my file that I want to load.

# ## Looking at the data <a id="looking_at_data"></a>
# In order to look at the data, you either just type `df` to have it print out the entire DataFrame, or you can use the `.head()` command. It's generally preferable to use `.head()`, since that just shows the first 5 rows (by default). If you *do* want to see more rows, you can pass a number into that function, telling it how many rows you want to see. I did that below to see 10 rows.

# In[56]:


df.head()


# In[57]:


df.head(10)


# Note that Pandas always includes an `index` row on the left, which uniquely numbers each row. We'll use these later.
# 
# What about if you want to access a single column? The way to do this is to pass the column name as a string (meaning in quotes) to the DataFrame.

# In[58]:


df['Age'].head()


# See how printing a single column looks different from printing multiple columns? That's because Pandas calls something with *a single column* a `Series`, and something with multiple columns a `DataFrame`. In reality, they're not that different. But it's helpful to know the terms in case you see people mentioning `Series` when you're Googling.

# If you want to access *multiple* columns, we need to pass those columns as a `list`. We'll pass to the DataFrame a `list` of the columns we're looking for. Remember that to access a column in Pandas we write `df[column_I_want]`. Since our columns are a list, they'll have brackets around them. This means we end up with double-brackets, like `df[[column1, column2, column3]]`.
# 
# Again, we *could* print everything we get back by typing `df[['Survived', 'Age']]`, but we probably just want a quick peek at the data, not to see the *entire* list of values. So I'll just look at the `.head()` of the data.

# In[59]:


df[['Survived', 'Age']].head()


# So far we've just accessed *columns*, so what about *rows*? Rows are accessed slightly differently. You use the `.iloc` command. Here "loc" refers to "location", and "i" refers to integer. So we'll access the row location by giving it an integer. As in everything in Python, we start at zero. So the first row we would access as `df.iloc[0]`, and the tenth row would be `df.iloc[9]`. Note that Pandas makes this easier for you by printing the index of each row as the first column (as discussed above). Let's try it below.

# In[60]:


df.iloc[0]


# Sure enough, if we look at when we first loaded the data, this Mr. Owen Harris Braund is indeed the first person listed, and has an `index` of zero.
# 
# We previously saw how to access multiple *columns*, so how do we access multiple *rows*? You can do it similar to how we accessed multiple columns, by passing a list of the row integers we want.

# In[61]:


df.iloc[[0, 1, 5]]


# However, suppose we want to access row 0 through 10000. Clearly we don't want to type `[0, 1, 2, ..., 10000]`. Instead, we can use **slicing**. Slicing is a shortcut way to say "give me everything between these two numbers". We "slice" by using a colon. 

# In[62]:


df.iloc[0:5]


# For reasons we won't go into now, Python decides that when you type `0:5`, you *don't* want `[0, 1, 2, 3, 4, 5]`, instead you must want `[0, 1, 2, 3, 4]` (so starting at 0, but ending just *before* 5). So the command above gave us the first five rows. Here's another simple one.

# In[63]:


df.iloc[10:20]


# You can also tell it to skip rows (like skip every other row, or just take every fifth row) by including that number at the end. So to get every fifth row between rows 0 and 20 you would type:

# In[64]:


df.iloc[0:20:5]


# ## Describing the data numerically <a id="data_numerically"></a>
# Okay, we've loaded our data, now let's start examining it. We probably want to answer questions like:
# - How many people survived? How many died?
# - Who is the oldest and youngest person on board? The average age?
# - For the people who are first class (`Pclass==1`), what percentage survived? What about for second and third class? 
# 
# We'll address each of these separately below.

# ### Value counts
# "Value counts" refers to counting how many different values appear in a column. So this could be used to answer the question of "How many people survived/died?", since I'm just counting how many 1's (survived) and 0's (died) appear in the `Survived` column. To do a value count, just select the column you want from your DataFrame (using what you learned above), and tell it to run the `.value_counts()` command.

# In[65]:


df['Survived'].value_counts()


# We can see that 549 people died, and 342 survived. Try it on your own for several other columns.

# ### Basic statistics
# Say you want to compute some basic statistics on a column, such as min/max/average/median/etc. In general, Pandas makes this very simple:

# In[66]:


df['Age'].min()


# In[67]:


df['Age'].max()


# In[68]:


df['Age'].mean()


# In[69]:


df['Age'].median()


# ## Analyzing subsets of data <a id="subset_data"></a>
# The last question about analyzing individual passenger classes is a little different. Up to this point we've just been taking an *entire column* (like the `Age` column) and finding something about it (`mean`, `median`, `value_counts`, etc). But what about if we just want a *subset* of the column, such as the first class passengers? To do this, you need to tell Pandas the *condition* you want satisfied. If we want first class passengers, we want the column `Pclass` to be equal to 1. Remember that in Python, to *test equality* (i.e. to check if something is or is not equal to something else) you use double equals signs `==`. So if I want to check for `Pclass` equal to 1, I would do `Pclass==1`. Let's see what happens if we try that.

# In[70]:


df['Pclass']==1


# Hmm, it looks like it's telling us `True` or `False` for whether each passenger was first class. That's not quite what I wanted, I just wanted the first class passengers. But that's okay, we're mostly there. Instead, what we do is say "DataFrame, look inside yourself and find the rows where `Pclass==1`".

# In[71]:


df[df['Pclass']==1].head()


# Now we've just got the first class passengers! Let's do a value count on just those passengers to see how many survived. We need to tell Pandas "Take this subset of the passengers which are first class, look at the `Survived` column, and do a `value_counts()`."

# In[72]:


df[df['Pclass']==1]['Survived'].value_counts()


# This is good, but I'm probably more interested in *percentages* than in raw numbers. To find out how to do this, Google "pandas value counts percentage". See if you can find the answer on your own. After you've done that, try comparing survival rates for second and third class passengers, as well as for all passengers regardless of class.

# ## Finding missing data (and dealing with it) <a id="missing_data"></a>
# It's very common that in a dataset, some of the data will be missing. What do we mean by "missing"? Sometimes it's just a blank cell in the spreadsheet, sometimes it's represented by `NaN`, which stands for "Not a number", and is Python speak for "I don't have anything useful to put here". There is also a `None` object in Python, which could potentially be in a cell. The cell could also have an empty string `''`, or a bunch of spaces `'   '`, or a zero where it doesn't make sense (like name), or $\pm\infty$, or one of many, many other things that could go wrong! This gives you some idea why finding and dealing with missing data is not simple. The general rule of thumb is that cleaning and prepping your data is about 80% of your work, while analyzing and utilizing it is only 20%.
# 
# Let's start by finding `None`s and `NaN`s. Pandas actually has a nice [helper function for this](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isna.html) called `isna()`. It looks for both `NaN` and `None` values. It *does not* look for spaces, zeros where you don't want them, etc. Let's try it with the `Cabin` column.

# In[73]:


df['Cabin'].isna()


# Hmm, that's not quite what we wanted. It's saying whether each row is or is not `NaN`/`None`. In a way that should look similar to what we had above with `df['Pclass'==1]`. We need to think about what our goal is. Is it just to *find* these values, to *drop rows with them*, or to *fix them*? Let's start with the first, just counting them.
# 
# The simplest way to count them is to see how many of the rows evalued to `True` when we did `.isna()`. A fantastically simple way to do this is below.

# In[74]:


df['Cabin'].isna().sum()


# What's going on here? The `.isna()` part is giving us the `True`/`False` values, and `.sum()` is adding them up. But how do we add `True` and `False`? These are what are called "boolean values", which mean that can be interpreted as either being 1 or 0. By default, `True` is mapped to 1, and `False` is mapped to 0. So when you call `.sum()`, it's adding up 1 (`True`) + 0 (`False`) + 1 (`True`) + ... and so forth. So the sum is exactly how many `NaN`/`None` values there are in that column!
# 
# How did I figure out this trick? Googling! I Googled "Pandas count number of nan none in column". The [first result](https://stackoverflow.com/questions/26266362/how-to-count-the-nan-values-in-a-column-in-pandas-dataframe/55608360) gave me a solution, which is what I used. You should get used to Googling for how to do things in programming. No human knows every command and trick, but the internet does! So use it to your advantage.
# 
# With that, let's count the number of `NaN`/`None` values are in each row. I'll use a simple `for`-loop to make this easier.

# In[75]:


for col in df.columns:
    print(f'{col}: {df[col].isna().sum()}')


# So it looks like most columns are pretty good, but some do indeed have a lot of missing values. What *percentage* of the values are missing? How about we divide the number of missing values by the number of rows. To get the shape of a DataFrame, use the `.shape` attribute.

# In[76]:


df.shape


# This says there are 891 rows (rows always come first, just like in a matrix) and 12 columns. Since we only care about rows, we'll grab the first part of this tuple by using `df.shape[0]` (remember that everything in Python is zero-indexed, so the *first* element in a tuple is referred to as the zeroth-element). Let's alter our `for`-loop above to print this.

# In[77]:


for col in df.columns:
    print(f'{col}: {df[col].isna().sum()} ({100 * df[col].isna().sum() / df.shape[0]}%)')


# How should we handle missing data? One way is just to ignore rows with missing data. We can do this by selecting rows where `.isna()` is `False`. Pandas has a helper function to do this, called `.dropna()`. By default it drops any row with at least one `NaN`/`None` value. But if you want more flexibility, you can set `.dropna(thresh=3)` (for instance) to only drop rows with *three* `NaN`/`None` values. It's good practice to always keep your original data intact. So once we drop the rows, we'll assign this to a new DataFrame called `df_trimmed`.

# In[78]:


df_trimmed = df.dropna()


# In[79]:


df_trimmed.shape


# How do we feel about this? We went from having at last *some* information on 891 passengers, to having *all* information on 183 passengers. This seems problematic, as I've thrown away a bunch of my data. Moreover, it seems like the column that most often has missing data is the `Cabin` column. Looking at that column, it seems like it's basically the room number for the person. I don't feel like that's so important, so maybe I don't need to worry about missing values there. Let's instead focus on the missing column which seems the most useful, which I feel is `Age`.
# 
# Rather than dropping rows with a missing `Age`, how about we fill in an age. This may seem strange, as I'm basically just making up an age for that person. However, soon we'll start building maching learning models, which use the data we have to make predictions about who will survive and die. For these models to work, they *need* to have data. If a row has a `NaN`/`None`, the model will throw an error and refuse to run. So it's better to have *something* so that it at least runs, than to have nothing.
# 
# The way to fill missing data is to use the Pandas helper function `.fillna()` (see a pattern here? Pandas has tons of helper functions. So before you try to write something from scratch, try Googling first!). All that `.fillna()` needs is an argument telling it *what* to fill in to the missing values. We *could* do something simple like filling in zeros (`.fillna(0)`), or 999, or -1, but it's generally best to not introduce extreme/unrealistic values (no one on board had an age of -1. Also, an age of 999 would skew the mean age, if that was something we decided to calculate). So good practice is to fill numerical columns with the median value.

# In[80]:


df_cleaned = df
df_cleaned['Age'] = df_cleaned['Age'].fillna(df_cleaned['Age'].median())


# Let's inspect our data and do another check for missing values to see where we're at.

# In[81]:


for col in df_cleaned.columns:
    print(f'{col}: {df_cleaned[col].isna().sum()} ({100 * df_cleaned[col].isna().sum() / df_cleaned.shape[0]}%)')


# In[82]:


df_cleaned.head()


# Looking good!
# 
# What about other cases, where the data is not what you would expect (numbers for `Name`, words for `Fare`, etc.)? One simple way is to look at the data type for each column. Pandas forces all values in a column to be the same type (`int`, `float`, `object`, etc.). So if a single value in the `Fare` column can't be described by a `float` (for example, it was a string '$123.16'), then Pandas will force that whole column to be strings (which Pandas calls an `object`).

# In[83]:


df_cleaned.dtypes


# A quick glance down the list looks good for the most part, but there is one weird one. `Age` is a float (decimal). Presumably all ages are integers, so we could consider changing that. The way to tell Pandas to change the column data type is using the `.astype()` helper function. As an argument, pass what type you would like. We'll use `int` as the type.

# In[84]:


df_cleaned['Age'] = df_cleaned['Age'].astype('int')


# In[85]:


df_cleaned.dtypes


# In[86]:


df_cleaned.head()


# ## Working with Series <a id="series"></a>
# 
# In Pandas, a Series is just a DataFrame with only one column. Unfortunately, this just presents a headache. There will be times when you do something which would work perfectly well with a DataFrame, but it will fail because you won't have realized that you only have one column and thus a Series!
# 
# Up to this point we have gotten our data by loading it from a CSV file. However, it's sometimes helpful to just make some quick, fake data to deal with. The easiest way to do this is to use the Pandas command `DataFrame()` and give it a dictionary. The dictionary should have the keys being the column names, and the values being a list of whatever values should be in that column. Here is an example.

# In[87]:


df = pd.DataFrame({'math_course_number': [2413, 3320, 3439],
                  'math_course_name': ['Calculus I', 'Applied Statistics', 'Introduction to Data Science']})

df


# This is a DataFrame. You can tell because it looks like one (you'll see what a Series looks like in a second), or by using the `type()` Python function.

# In[88]:


type(df)


# Now let's select just a single column and see what we get.

# In[89]:


math_course_number = df['math_course_number']

math_course_number


# This looks different! It doesn't have the nice pretty table-style look to it. That's because this isn't a DataFrame, it's a Series!

# In[90]:


type(math_course_number)


# Some things work just as well using a Series as it does a DataFrame. For example:

# In[91]:


math_course_number.head()


# In[92]:


math_course_number + 1


# However, some things don't!

# In[93]:


# An error is coming...
math_course_number['math_course_number']


# Series don't give the columns names, so you can't access them by name anymore. In general, it's often just easiest to convert a Series back to a DataFrame, then you are always working with the same objects. There are (at least) two ways to do this:
# 1. Use the Pandas `.to_frame()` function
# 2. Use the Pandas `.reset_index()` function
# 
# These two do different things, so let's briefly talk about each.
# 
# The `.to_frame()` method simply turns a Series into a DataFrame, that's it. Here's an example.

# In[94]:


math_course_number_df = math_course_number.to_frame()

math_course_number_df


# Now it looks like (and is!) a DataFrame again!

# In[95]:


type(math_course_number_df)


# The only option you have with `.to_frame()` is that you can specify the column name. If, instead of 'math_course_number' you wanted something else, you could do that.

# In[96]:


math_course_number_df = math_course_number.to_frame(name='course_number')

math_course_number_df


# Now let's talk about `.reset_index()`. What this function does is it makes the index (the number on the left of the Series/DataFrame) become a new column. Here's an example:

# In[97]:


math_course_number


# In[98]:


math_course_number.reset_index()


# You can see that we have a new column called "index" which has the numbers on the left. This can often be extremely useful. You will run into situations where you have an index having numbers you want to use, but since they're not a column you can't easily access them. In that case, use `.reset_index()` to make them into a column.

# ## Exercises
# 
# While you can use these to get you started, really the best thing you can do is to just mess around with the data and come up with your own questions. Then start exploring those questions. Undoubtedly you will run into things you can't figure out. Just Google and see what you can come up with. Patience and perserverance are key!
# 
# 1. For each passenger class (`Pclass`), what are the cheapest, most expensive, mean and median prices?
# 2. What is the average age for men and the average age for women?
# 3. The `SibSp` column counts how many siblings and spouses are on board for that person (so if you and your sister on the ship together, then this would be 1, because you have one sibling or spouse). Create a new DataFrame containing only the people with a sibling or spouse on board.
# 1. The cabin column is probably are not useful in any way, especially with so many missing values. Drop that column from the data (Google how to do it!).
# 6. What patterns do you see about survival rates among passengers in different classes? We looked at just first-class passengers, but what about second and third class?
# 7. We saw above that about 63% of first-class passengers survived, this is far better than for second or third class passengers. But can you do better? What about first-class females? Or second-class children? Can you find some subset of the people with a very strong survival rate? What about a high death rate?
# 8. So far we've been asking about the relationship between just two variables: `Age` and `Survived`, `Embarked` and something else, etc. How would you go about exploring the relationship between *three or more* variables? Perhaps `Survived` is dependent on `Age`, `Fare`, `Sex`, `SibSp` and `Parch`. How could you explore this?
# 9. We described "missing data" data with a `None` value. However, that's not the only way for data to be missing. An empty string `""` would also be "missing". Finally, while data may not be missing per-se, it can still be a value you don't want, such as a `Sex` of "unknown". Checking each column to see if there are any values that you don't want. Did you find any?

# ## Resources
# 
# Where are some places you can turn when you have questions? Google is obviously one of them, but here are some other sites I've found especially useful:
# 
# - [Chris Albon](https://chrisalbon.com/#python) - He has very clear explanations for lots of different things. He pretty much always starts by creating a DataFrame from scratch, and then works through simple examples. A great first resource if Google isn't working for you.
# - [Machine Learning Mastery](https://machinelearningmastery.com/start-here/) - A site covering more advanced topics, but still in a very simple, readable format.
# - [Kaggle kernels](https://www.kaggle.com/kernels) - "Kernels" is Kaggle's word for Jupyter notebooks. They host notebooks on their website, and people will build them to accomplish various tasks, and then share them with the community. Many of them are like this notebook, where they're tutorials meant to take you from start-to-finish on a task. Less useful for answering a quick "how do I..." question, but highly useful for finding an introduction to a new topic.

# In[ ]:





# 
# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# ch_02
# ch_03
# ch_04
# ch_05
# ch_06
# ch_07
# ch_08
# ch_09
# ch_10
# ch_11
# ```
# 
