import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from random import shuffle



def get_data():
    tourd=pd.read_excel("dataset/data-ver-6.xlsx",index_col=0)
    tourd[['category','big_description']]=tourd[['category','big_description']].fillna('')
    tourd['image'] = tourd['image'].fillna('https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/No-Image-Placeholder.svg/1200px-No-Image-Placeholder.svg.png')
    tourd['imageSet'] = tourd['imageSet'].fillna('https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/No-Image-Placeholder.svg/1200px-No-Image-Placeholder.svg.png')

    return tourd

def transform_data(data):
    count = CountVectorizer(stop_words='english')
    count_matrix1 = count.fit_transform(data['category'])
    count_matrix2 = count.fit_transform(data['state'])
    
    combine_sparse = sp.hstack([count_matrix1,count_matrix2], format='csr')
    cosine_sim = cosine_similarity(combine_sparse, combine_sparse)
    
    return cosine_sim

def recommend_destinations(title, data, transform):
    indices = pd.Series(data.index, index = data['name'])
    index = indices[title]
    
    sim_scores = list(enumerate(transform[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]

    dest_indices = [i[0] for i in sim_scores]

    dest_state = data['address'].iloc[dest_indices]
    dest_name = data['name'].iloc[dest_indices]
    dest_desc = data['big_description'].iloc[dest_indices]
    image = data['image'].iloc[dest_indices]
    imageSet = data['imageSet'].iloc[dest_indices]

    recommendation_data = pd.DataFrame(columns=['name','location', 'details','img','imgSet'])

    recommendation_data['location'] = dest_state
    recommendation_data['name'] = dest_name
    recommendation_data['details'] = dest_desc
    recommendation_data['img'] = image
    recommendation_data['imgSet'] = imageSet

    return recommendation_data

def results(destination):
    data = get_data()
        
    transform_result = transform_data(data)

    if destination not in data['name'].unique():
        return "[]"

    else:
        recommendations = recommend_destinations(destination, data, transform_result)
        df= recommendations.to_dict('records')
        for i in range(len(df)):
            df[i]["imgSet"] = list((df[i]["imgSet"]).split("[#]"))
        return df
