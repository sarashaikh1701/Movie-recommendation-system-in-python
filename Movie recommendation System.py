#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import difflib  #close match for movie name entered
from sklearn.feature_extraction.text import TfidfVectorizer  #covert text to numerical// feature vectors
from sklearn.metrics.pairwise import cosine_similarity


# In[22]:


df = pd.read_csv('movies.csv')
df.head(8)


# In[23]:


df.shape


# In[24]:


df.columns


# removing irrelevant data:

# In[25]:


selected_features=['genres', 'keywords', 'tagline', 'cast', 'director']
selected_features


# replacing na values

# In[26]:


for feature in selected_features:
    df[feature]= df[feature].fillna('')  #filling na values with null strings


# # combining all relevant features

# In[27]:


combined_features=df['genres']+' '+df['keywords']+' '+df['tagline']+' '+df['cast']+' '+df['director']
combined_features


# # converting text data to numerical values: featured vectors

# In[28]:


vectorizer = TfidfVectorizer() #loading into vectorizer
feature_vector= vectorizer.fit_transform(combined_features) #storing all num values
feature_vector


# # using cosine similarity(similarity score)

# In[29]:


similarity= cosine_similarity(feature_vector) #find similarity from perviously converted num values
similarity


# In[30]:


similarity.shape


# # getting movie name from user

# In[31]:


movie_name=input('Enter your favourite movie name: ')


# # creating list with all movie names in ds

# In[40]:


list_of_titles= df['original_title'].tolist()
list_of_titles


# # finding close match for user movie

# In[33]:


find_close_match= difflib.get_close_matches(movie_name, list_of_titles)  #The difflib module is generally used to compare the sequence of the strings. But we can also use it to compare other data types as long as they are hash-able.
find_close_match


# In[34]:


close_match=find_close_match[0]
close_match


# # finding index of movie using title

# In[35]:


index= df[df.original_title == close_match]['index'].values[0] #in form of a list
index


# # getting list of similar movies

# In[36]:


simi_score= list(enumerate(similarity[index]))
simi_score
#first value gives index and second gives similarity score


# # sorting movies with highest similarity scores

# In[37]:


sorted_simi_movies=sorted(simi_score, key = lambda x:x[1], reverse = True)
sorted_simi_movies
# x is simi score and x[1] is value of it, arraging in descending order


# # printing similar movies using index

# In[38]:


print('Movies suggested for you')

i=1
for movie in sorted_simi_movies:
    index = movie[0]  
# movie[0] represents the first row of values in the sorted_simi_movies
    title_from_index = df[df.index==index]['original_title'].values[0]
    
    if (i<31):
        print(i, '.' , title_from_index)
        i=i+1


# In[ ]:





# # ACTUAL SYSTEM

# In[43]:


movie_name=input('Enter your favourite movie name: ')

list_of_titles= df['original_title'].tolist()

find_close_match= difflib.get_close_matches(movie_name, list_of_titles)

close_match=find_close_match[0]

index= df[df.original_title == close_match]['index'].values[0]

simi_score= list(enumerate(similarity[index]))

sorted_simi_movies=sorted(simi_score, key = lambda x:x[1], reverse = True)

print('Movies suggested for you')

i=1
for movie in sorted_simi_movies:
    index = movie[0]  
# movie[0] represents the first row of values in the sorted_simi_movies
    title_from_index = df[df.index==index]['original_title'].values[0]
    
    if (i<31):
        print(i, '.' , title_from_index)
        i=i+1


# This system can be further modified according to the user input. For example, a different system can be created for a user that wants recommendations based on the actor as input, etc. This recommendation system will keep building in future as new movies will keep releasing and requirements of users will change.

# In[ ]:




