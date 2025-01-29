from django.shortcuts import render
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


mv_d = pd.read_csv('movies.csv')
s_fea = mv_d[['genres', 'keywords', 'tagline', 'cast', 'director']]


for f in s_fea:
    mv_d[f] = mv_d[f].fillna('')


com = mv_d['genres'] + ' ' + mv_d['keywords'] + ' ' + mv_d['tagline'] + ' ' + mv_d['cast'] + ' ' + mv_d['director']


vectorizer = TfidfVectorizer()
f_vectors = vectorizer.fit_transform(com)


sim = cosine_similarity(f_vectors)


list_all = mv_d['title'].tolist()

def home(request):
    if request.method == 'POST':
        mv_m = request.POST.get('movie name')  
        
        #
        find = difflib.get_close_matches(mv_m, list_all)
        
        if find:  
            cmatch = find[0]

            
            index_m = mv_d[mv_d.title == cmatch]['index'].values[0]

            
            sim_score = list(enumerate(sim[index_m]))

            
            sort_sim_m = sorted(sim_score, key=lambda x: x[1], reverse=True)

            
            title = []
            rate = []
            tim = []

            i = 1
            for movie in sort_sim_m:
                index = movie[0]
                title_from_index = mv_d[mv_d.index == index]['title'].values[0]
                rating = mv_d[mv_d.index == index]['vote_average'].values[0]
                time = mv_d[mv_d.index == index]['runtime'].values[0]

                if i < 10:  
                    title.append(title_from_index)
                    rate.append(rating)
                    tim.append(time)
                    i += 1

            
            movie_data = zip(title, rate, tim)

            
            content = {
                'movie_data': movie_data
            }

            return render(request, 'home.html', content)

    
    return render(request, 'home.html')
