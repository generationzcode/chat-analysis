
import math

import datetime
import nltk
from nltk import word_tokenize, pos_tag
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
from nltk.corpus import cmudict
import re
nltk.download('cmudict')

"""# Preprocessing the data"""
is_apple = True
# apple way
if is_apple:
    f = open("filename.txt").read().split("\n")[2:]
    date_used = False
    init_date = 0
    for i,v in enumerate(f):
      if "\u200e" in f[i]:
        f[i] = f[i].replace("\u200e","")
      if "\u202f" in f[i]:
        f[i] = f[i].replace("\u202f","")
      if "\u200d" in f[i]:
        f[i] = f[i].replace("\u200d","")
      if "\u2060" in f[i]:
        f[i] = f[i].replace("\u2060","")
      if "\u200c" in f[i]:
        f[i] = f[i].replace("\u200c","")

    message_added = []#this removes the effect of newlines in messages from breaking them up
    for i in f:
      try:
        if i[0]!="[":
          message_added[-1]+=i
        else:
          message_added.append(i)
      except:
        pass
    def get_date(message):
      global init_date
      global date_used
      if date_used:
        return (datetime.datetime.strptime(message[message.index("[")+1:message.index("]")], "%d/%m/%y, %I:%M:%S%p")-init_date).seconds/60
      date_used = True
      init_date = datetime.datetime.strptime(message[message.index("[")+1:message.index("]")], "%d/%m/%y, %I:%M:%S%p")
      return 0
    person_dict = {}
    for i in message_added:
      message_start = i[i.index("]")+2:]
      if message_start[:message_start.index(":")] not in person_dict:
        person_dict[message_start[:message_start.index(":")]] = [[get_date(i),message_start[message_start.index(":")+2:]]]
      else:
        person_dict[message_start[:message_start.index(":")]].append([get_date(i),message_start[message_start.index(":")+2:]])
else:
# android way
    f = open("/content/WhatsApp Chat with ICT BCHEM UNOFFICIAL (2).txt").read().split("\n")[2:]
    date_used = False
    init_date = 0
    for i,v in enumerate(f):
      if "\u200e" in f[i]:
        f[i] = f[i].replace("\u200e","")
      if "\u202f" in f[i]:
        f[i] = f[i].replace("\u202f","")
      if "\u200d" in f[i]:
        f[i] = f[i].replace("\u200d","")
      if "\u2060" in f[i]:
        f[i] = f[i].replace("\u2060","")
      if "\u200c" in f[i]:
        f[i] = f[i].replace("\u200c","")

    message_added = []#this removes the effect of newlines in messages from breaking them up
    for i in f:
      try:
        try:
          datetime.datetime.strptime(i[:i.index("-")-1], "%d/%m/%y, %I:%M%p")
          message_added.append(i)
        except:
          message_added[-1]+=i
      except:
        pass
    print(message_added)
    def get_date(message):
      global init_date
      global date_used
      if date_used:
        return (datetime.datetime.strptime(message[:message.index("-")-1], "%d/%m/%y, %I:%M%p")-init_date).seconds/60
      date_used = True
      init_date = datetime.datetime.strptime(message[:message.index("-")-1], "%d/%m/%y, %I:%M%p")
      return 0
    person_dict = {}
    for i in message_added:
      try:
        message_start = i[i.index("-")+2:]
        if message_start[:message_start.index(":")] not in person_dict:
          person_dict[message_start[:message_start.index(":")]] = [[get_date(i),message_start[message_start.index(":")+2:]]]
        else:
          person_dict[message_start[:message_start.index(":")]].append([get_date(i),message_start[message_start.index(":")+2:]])
      except:
        pass

"""# Generating filter scores
* word frequency function, time dependant
* sentiment function, total
* POS function, time_dependant - stylometric analysis
* readability function, total

time dependant functions will be regular except:
$$\frac{score}{\Delta t} = new\space score$$

## POS distribution vector cosine similarity rank

This code is too heavy to rank everyone, so pick one person and see:
"""

person_dict.keys()

POI = "Yogiraj Jadhav"

"""### Scoring function:"""

def preprocess(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return [tag for word, tag in pos_tags]
def calculate_pos_distribution(pos_tags):
    pos_counts = nltk.FreqDist(pos_tags)
    total_tags = len(pos_tags)
    pos_distribution = {tag: count / total_tags for tag, count in pos_counts.items()}
    return pos_distribution
def create_pos_vector(pos_distribution, all_pos_tags):
    vector = [pos_distribution.get(tag, 0) for tag in all_pos_tags]
    return np.array(vector)
def pos_similarity_score(text1, text2):
    pos_tags1 = preprocess(text1)
    pos_tags2 = preprocess(text2)
    pos_dist1 = calculate_pos_distribution(pos_tags1)
    pos_dist2 = calculate_pos_distribution(pos_tags2)
    all_pos_tags = list(set(pos_dist1.keys()).union(set(pos_dist2.keys())))
    vector1 = create_pos_vector(pos_dist1, all_pos_tags)
    vector2 = create_pos_vector(pos_dist2, all_pos_tags)
    similarity = cosine_similarity([vector1], [vector2])[0][0]
    return similarity

"""### Total ranking:"""

# This was time dependant but it took too much processing
"""
rank_dict = {}
i=POI
POS_dict = {}
for m in person_dict:
  POS_dict[m] = 0
for v in person_dict:
  if i!=v:
    for x in person_dict[i]:
      for l in person_dict[v]:
        try:
          POS_dict[v]+=math.sqrt((pos_similarity_score(x[1],l[1])/(x[0]-l[0]))**2)/len(person_dict[v])
        except:
          pass
rank_dict[i] = POS_dict
rank_dict
"""
rank_dict = {}
i=POI
POS_dict = {}
messages_only_dict = {}
for i in person_dict:
  messages_only_dict[i] = []
  for v in person_dict[i]:
    messages_only_dict[i].append(v[1])
for i in person_dict:
  for m in person_dict:
    POS_dict[m] = 0
  for v in person_dict:
    if i!=v:
      POS_dict[v] = pos_similarity_score(".".join(messages_only_dict[i]),".".join(messages_only_dict[v]))
  rank_dict[i] = POS_dict
rank_dict

"""## Word frequency rank"""

nltk.download('stopwords')
from nltk.corpus import stopwords
stops = list(stopwords.words('english'))

people_message_words  = {}
messages_only_dict = {}
for i in person_dict:
  messages_only_dict[i] = []
  for v in person_dict[i]:
    messages_only_dict[i].append(v[1])
#people_num_messages['Ishan Joshi Gangsta W/A Camera'] = ['a']
#people_num_messages['1'] = ['a']
for i in messages_only_dict:
  people_message_words[i] = []
  for n,v in enumerate(messages_only_dict[i]):
    words = v.split(" ")
    wordsFiltered = [w for w in words if w not in stops]
    for m in wordsFiltered:
      if m not in people_message_words[i]:
        people_message_words[i].append([person_dict[i][n][0],m])
pos_gen = {}
for p in people_message_words:
  rank_gen = {}
  for v in people_message_words:
    rank_gen[v] = 0
    for i in people_message_words[p]:
      if i!=v:
        for m in people_message_words[v]:
          if i[1]==m[1]:
            try:
              rank_gen[v]+=math.sqrt(1/(len(people_message_words[v])*(i[0]-m[0]))**2)
            except:
              pass
  pos_gen[p] = rank_gen
pos_gen

"""## Sentiment analysis"""

sentiment_dict = {}
for i in person_dict:
  weird = 0
  for m,v in enumerate(person_dict[i]):
    weird += sia.polarity_scores(v[1])['compound']
  sentiment_dict[i] = weird
sentiment_dict
#sia.polarity_scores(v)['compound']

visualizing_positivity = {}
sentiment_array = []
for i in sentiment_dict:
  sentiment_array.append(sentiment_dict[i])
v_pos = np.percentile(sentiment_array,90)
pos = np.percentile(sentiment_array,60)
neu = np.percentile(sentiment_array,40)
neg = np.percentile(sentiment_array,20)
v_neg = np.percentile(sentiment_array,0)
# this will loop over sentiment_dict and list how positive someone is according to the percentiles
for i in sentiment_dict:
  if sentiment_dict[i] > v_pos:
    visualizing_positivity[i] = "very positive"
  elif sentiment_dict[i] > pos:
    visualizing_positivity[i] = "positive"
  elif sentiment_dict[i] > neu:
    visualizing_positivity[i] = "neutral"
  elif sentiment_dict[i] > neg:
    visualizing_positivity[i] = "negative"
  else:
    visualizing_positivity[i] = "depressed"
visualizing_positivity



"""## readability index

"""

def flesch_kincaid_grade(text):
    num_sentences, num_words, syllable_count, _ = analyze_text(text)
    score = 0.39 * (num_words / num_sentences) + 11.8 * (syllable_count / num_words) - 15.59
    return score
cmu_dict = cmudict.dict()
def count_syllables(word):
    word = word.lower()
    if word in cmu_dict:
        return [len(list(y for y in x if y[-1].isdigit())) for x in cmu_dict[word]][0]
    else:
        return len(re.findall(r'[aeiouy]+', word))
def analyze_text(text):
    sentences = nltk.sent_tokenize(text)
    num_sentences = len(sentences)
    words = nltk.word_tokenize(text)
    num_words = len(words)
    syllable_count = sum(count_syllables(word) for word in words)
    complex_words_count = sum(1 for word in words if count_syllables(word) >= 3)
    return num_sentences, num_words, syllable_count, complex_words_count

readability_dict = {}
for i in person_dict:
  for v in messages_only_dict[i]:
    try:
      readability_dict[i] += flesch_kincaid_grade(v)/len(messages_only_dict[i])
    except:
      readability_dict[i] = flesch_kincaid_grade(v)/len(messages_only_dict[i])
readability_dict



"""*italicized text*# final ranking

## Friend Ranking
"""

# rank_dict, pos_gen, sentiment_dict, readability_dict
#pos analysis
rank_fac = 1
big_rank = 0
for i in rank_dict:
  arr = []
  for v in rank_dict[i]:
    arr.append(rank_dict[i][v])
  big_rank = max(sum(arr)/len(arr),big_rank)
rank_fac = 1/big_rank
#word freq analysis
pos_fac = 1
big_rank = 0
for i in pos_gen:
  arr = []
  for v in pos_gen[i]:
    arr.append(pos_gen[i][v])
  big_rank =max(sum(arr)/len(arr),big_rank)
pos_fac = 1/big_rank
# sentiment analysis
sum_of_all = 0
for i in sentiment_dict:
  sum_of_all+=sentiment_dict[i]
sentiment_fac = len(sentiment_dict)/sum_of_all
# readability analysis
sum_of_all = 0
for i in readability_dict:
  sum_of_all+=readability_dict[i]
readability_fac = len(readability_dict)/sum_of_all
total_dict = {}
for i in rank_dict:
  total_dict[i]={}
for i in rank_dict:
  for v in rank_dict[i]:
    if i!=v:
      try:
        total_dict[i][v] = rank_dict[i][v]*rank_fac+pos_gen[i][v]*pos_fac+smath.sqrt((1/(sentiment_dict[i]*sentiment_fac-sentiment_dict[v]*sentiment_fac))**2)+math.sqrt((1/(readability_dict[i]*readability_fac-readability_dict[v]*readability_fac))**2)
      except:
        total_dict[i][v] = rank_dict[i][v]*rank_fac+pos_gen[i][v]*pos_fac+math.sqrt((1/(sentiment_dict[i]*sentiment_fac-sentiment_dict[v]*sentiment_fac+0.1))**2)+math.sqrt((1/(readability_dict[i]*readability_fac-readability_dict[v]*readability_fac+0.1))**2)*readability_fac
    else:
      total_dict[i][v] = 0

for i in total_dict:
  total_dict[i] = dict(sorted(total_dict[i].items(), key=lambda item: item[1],reverse=True))
for i in total_dict:
  count = 0
  print("\n")
  print(i)
  for v in total_dict[i]:
    if count>5:
      break
    print("     "+v)
    count+=1

rg_friends = {}
for i in total_dict:
  rg_friends[i] = {}
  for v in total_dict:
    rg_friends[i][v] = 0
    if i!=v:
      for m in total_dict:
        in1 = 0
        in2 = 0
        for d,l in enumerate(total_dict[m]):
          if l==i:
            in1 = d
          if l==v:
            in2 = d
        rg_friends[i][v] += math.sqrt(1/((in1-in2)**2))
for i in rg_friends:
  rg_friends[i] = dict(sorted(rg_friends[i].items(), key=lambda item: item[1],reverse=True))
for i in rg_friends:
  count = 0
  print("\n")
  print(i)
  for v in rg_friends[i]:
    if count>5:
      break
    print("     "+v)
    count+=1

