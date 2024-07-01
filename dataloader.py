import pandas as pd
import gzip
import json
from nltk import sent_tokenize
import random

random.seed(42)

class Helpful_Review_Data:
    def __init__(self, path) -> None:
        data = [json.loads(line) for line in open(f'{path}/train.json', 'r')]
        df = pd.DataFrame(data)
        df = df[df['helpful']>=1.5]
        all_asins = set(df['asin'].tolist())
        all_revs,all_prods, all_ids = [],[],[]
        for asin in all_asins:
            all_prods.append(df[df['asin']==asin]['product_title'].tolist()[0])
            all_ids.append(asin)
            #print(asin)
            sentences = " ".join(x for x in df[df['asin']==asin]['sentence'].tolist())
            all_revs.append(sentences)
            #print(sentences)
        self.df_all = pd.DataFrame({'id':all_ids, 'titles':all_prods, 'reviews': all_revs})
    
    def get_data(self):
        return self.df_all



class Product_Data:
    def __init__(self, path, category):
        data = []
        self.df = pd.read_csv(f"{path}/data/{category}.txt", sep='\t', header=None)
        self.df.columns =['asin','pos_reviews','neg_reviews']
        pos_revs = self.df['pos_reviews'].tolist()
        neg_revs = self.df['neg_reviews'].tolist()
        with gzip.open(f'{path}/metadata/meta_{category}.json.gz') as f:
            for l in f:
                data.append(json.loads(l.strip()))
        self.asin_dict ={}
        for item in data:
            if "title" in item.keys():
                self.asin_dict[item["asin"]]=item["title"]

        product_names = []       
        for idx, rows in self.df.iterrows():
            if rows["asin"] in self.asin_dict:
                product_names.append(self.asin_dict[rows["asin"]])
            else:
                product_names.append("None")
        self.df["product_names"]=product_names
        self.df = self.df[self.df["product_names"]!="None"]

    def get_data(self):
        return self.df

    def get_product_names(self):
        product_names = self.df["product_names"].tolist()
        return product_names 
       
class DebateData:
    def __init__(self,path):
        path = 'iq2_data_release.json'
        with open(path, "r") as f:
            self.debates = json.load(f)

    def get_item(self, topic):
        data = self.debates[topic]
        for_args=[]
        against_args =[]
        for content in data['transcript']:
            if content['speakertype']=='for':
                for_args.append(content['paragraphs'][0])
            if  content['speakertype']=='against':
                against_args.append(content['paragraphs'][0])
        return for_args, against_args, data['title']

    def get_data(self):
        topics = self.debates.keys()
        all_for_args, all_against_args, all_topics = [],[],[]
        for topic in topics:
            for_args, against_args, title = self.get_item(topic)
            all_for_args.append(for_args[:2]) # take only the first argument
            all_against_args.append(against_args[:2])
            all_topics.append(title)
        
        idxs =  list(range(len(all_topics)))
        df = pd.DataFrame({'ID': idxs, 'topic': all_topics, 'for_args': all_for_args, 'against_args': all_against_args})
        print(df)
        return df
        


class MetaReview:
    def __init__(self, path):
        self.df = pd.read_csv(path, sep='\t', header=None)
        self.df.columns = ['ID', 'Paper', 'Type', 'Review 1', 'Review 2', 'Review 3']
        print(self.df)
    
    def get_data(self):
        df_comp = self.df[['ID', 'Paper', 'Type', 'Review 1', 'Review 2', 'Review 3']]
        df_comp = df_comp[df_comp['ID'].notna()]
        df_comp = df_comp[df_comp['Review 2'].notna()]
        df_comp = df_comp[df_comp['Review 3'].notna()]
        return df_comp



def main():
    helpful_data = Helpful_Review_Data("/src/data/helpful_sentences_reviews").get_data()
    debate_data = DebateData("src/data/iq2_data_release.json").get_data()
    meta_review_data = MetaReview("src/data/meta_reviews.tsv").get_data()
    print('Data Loaded!')

    helpful_data = helpful_data.head(100)
    debate_data = debate_data.head(100)
    meta_review_data = meta_review_data.head(100)

    helpful_data.to_csv("data/helpful_reviews.tsv", index=False, sep='\t')
    debate_data.to_csv("data/debates.tsv", index=False, sep='\t')
    meta_review_data.to_csv("data/meta_reviews.tsv", index=False, sep='\t')

if __name__=='__main__':
    main()