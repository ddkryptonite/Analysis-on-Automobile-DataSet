#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

df = pd.read_csv(url, header = None)


# In[ ]:





# In[2]:


df.head(5)
df.head(10)


# In[3]:


df.tail(5)


# In[2]:


df.dtypes


# In[5]:


df.dtypes


# In[6]:


df.describe()


# In[7]:


df.describe(include="all")


# In[8]:


df.info()


# In[3]:


headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers/n", headers)


# In[4]:


df.columns  = headers
df.columns


# In[11]:


df.head(10)


# In[5]:


import numpy as np

# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
df.head(5)

df1=df.replace('?',np.NaN)
df.head(10)


missing_data = df.isnull()
missing_data.head()

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    


# In[27]:


# Convert the 'normalized-losses' column to numeric, coercing errors to NaN
df.loc[:, "normalized-losses"] = pd.to_numeric(df["normalized-losses"], errors='coerce')

#Calculating mean to use for replacement
avg_norm_losses = df["normalized-losses"].astype(float).mean(axis=0)
print("Average of normalized-losses", avg_norm_losses)

# Replace NaN values with the calculated mean using .loc
df.loc[:, "normalized-losses"] = df["normalized-losses"].fillna(avg_norm_losses)

# Bore
df.loc[:, "bore"] = pd.to_numeric(df["bore"], errors='coerce')
avg_bore = df["bore"].mean()
print("Average of bore:", avg_bore)
df.loc[:, "bore"] = df["bore"].fillna(avg_bore)

# Stroke
df.loc[:, "stroke"] = pd.to_numeric(df["stroke"], errors='coerce')
avg_stroke = df["stroke"].mean()
print("Average of stroke:", avg_stroke)
df.loc[:, "stroke"] = df["stroke"].fillna(avg_stroke)

# Horsepower
df.loc[:, "horsepower"] = pd.to_numeric(df["horsepower"], errors='coerce')
avg_horsepower = df["horsepower"].mean()
print("Average of horsepower:", avg_horsepower)
df.loc[:, "horsepower"] = df["horsepower"].fillna(avg_horsepower)

# Peak RPM
df.loc[:, "peak-rpm"] = pd.to_numeric(df["peak-rpm"], errors='coerce')
avg_peak_rpm = df["peak-rpm"].mean()
print("Average of peak-rpm:", avg_peak_rpm)
df.loc[:, "peak-rpm"] = df["peak-rpm"].fillna(avg_peak_rpm)

df.head(20)


# In[24]:


df=df1.dropna(subset=["price"], axis=0)


# In[28]:


df.head(20)


# In[11]:


print(df.columns)


# In[7]:


df.to_csv("automobile.csv", index=False)


# In[29]:


df[['normalized-losses', 'make']].describe()


# In[16]:


df[['length', 'compression-ratio']].describe()


# In[50]:


df.head(10)


# In[51]:


# Rename column "city-mpg" to "city-L/100km"
#df.loc[:, "city-L/100km"] = df["city-mpg"]

# Perform conversion and renaming in one step
#df.loc[:, "city-L/100km"] = 235 / df["city-mpg"]

# Drop the original "city-mpg" column
#df = df.drop(columns=["city-mpg"])

# Display the updated DataFrame
print(df)

df["price"] = df["price"].astype("int")

df["length"] = df["length"]/df["length"].max()


# In[52]:


#FINDING CORRELATION BETWEEN COLUMNS

df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()


# In[65]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)



# In[ ]:




