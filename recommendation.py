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
    sim_scores = list(transform[index])
    for i in range(len(sim_scores)):
      sim_scores[i]=[i,sim_scores[i]]
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:81]


    for i in range(len(sim_scores)):
      sim_scores[i][1] = sim_scores[i][1] * data['rating'].iloc[sim_scores[i][0]] * data['numberOfRating'].iloc[sim_scores[i][0]]

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

def results(category):
    data = get_data()
    cd={'Temples': 'temple,hindu,mandir,shiva,gurudwara,sikh,jain,dham,mahadev,gurdwara', 'Religious': 'temple,mosque,mandir,shrine,church,monastery,pilgrimage,lord,worship,catholic,gurudwara', 'Buddhism': 'shrine,monastery,buddha,buddhism', 'Mughal': 'mosque,mughal,tomb,masjid,muslim,mausoleum,dome,domed,jama', 'Parks and Gardens': 'park,garden,fountain,playground,aquarium', 'Entertainment': 'lake,scenic,dam,hill,cave,bird,hilltop,zoo,amusement,shopping,dining,dating', 'Museums and Art Galleries': 'museum,gallery,art,antique', 'Monuments': 'monument,memorial,stupa,heritage', 'Heritage': 'iconic,unique,heritage', 'Beaches and Islands': 'beach,island,sea,ocean,sandy', 'History and Architecture': 'history,architecture,ancient,historic,historical,artifact', 'Nature': 'waterfall,scenic,water,mountain,river,nature,fall,valley,cave', 'Forts and Palaces': 'fort,palace,fortress,mahal,royal,qila,raj', 'Wildlife Sanctuaries': 'wildlife,sanctuary,national,forest,tiger,animal,forested,wild', 'Sports': 'sport,swimming,boating,hiking,trek,trekking'}
    df = {'name':'curr_user','category':cd[category],'popularity':'','ratingClass':'','state':''	}
    data=data.append(df,ignore_index=True)
        
    transform_result = transform_data(data)

    if category not in cd.keys():
        return None

    else:
        recommendations = recommend_destinations("curr_user", data, transform_result)
        return recommendations.to_dict('records')

def combine_results(cat_list):
  cat_list=list(cat_list.split(","))
  noe=20//len(cat_list)
  finalrec=list()
  for i in cat_list:
    rec=results(i)
    if rec != None:
      if len(cat_list)>1:
        rec=rec[0:10]
      shuffle(rec)
      j,k=0,0
      while j<noe and k<noe:
        if rec[k] not in finalrec: 
          finalrec.append(rec[k])
          k+=1
          j+=1
        else:
          k+=1
  shuffle(finalrec)
  for i in range(len(finalrec)):
    finalrec[i]['imgSet'] = list((finalrec[i]['imgSet']).split("[#]"))
  return finalrec
