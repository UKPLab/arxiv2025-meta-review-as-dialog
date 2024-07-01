import pandas as pd
from collections import defaultdict, Counter
import argparse, os
import numpy as np

domains =['debates', 'helpful_reviews', 'meta_reviews']
seeker_map = {"meta_reviews": "meta reviewer", "helpful_reviews": "buyer", "debates": "decision maker"}
wizard_name = ["dialogue agent"]


def preprocessing(x):
    return x.lower().replace("-"," ")

metric_map={"kbert":'f1', "kprecision": "kprecision", "q2_f1": "f1", "q2_nli": "nli", "faith_dial_critic":"faithdial_scores", "faith_dial_critic_labels": "scores", "specificity": "specificty"}


def score_dist(df, metric):
    col_name = metric_map[metric]
    if metric == 'faith_dial_critic_labels':
        c = Counter(df[col_name].to_list())
        if 'Hallucination' in c:
            return c['Hallucination']/sum(c.values())
        return 0
    scores = df[col_name].apply(lambda x: float(x))
    scores = scores.dropna()
    #print(scores)
    return scores.mean()

def avg_tokens_wizard(df, wizard_names, metric):
    wizard_names =[x for x in wizard_names if preprocessing(x) in wizard_name]
    wizard = df[df['speaker'].isin(wizard_names)]
    return score_dist(wizard, metric)


def avg_tokens_information_seeker(df, seeker_names, domain, metric):
    seeker_names =[x for x in seeker_names if preprocessing(x)== seeker_map[domain]]
    seeker = df[df['speaker'].isin(seeker_names)]
    return score_dist(seeker, metric)



def main(args):
    df = pd.read_csv(args.data_path, sep='\t')
    df = df.dropna()
    df.columns = df.columns.str.replace(' ', '')
    f = open(f'{args.output_path}/model_level_metrics_{args.model}_{args.metric}.txt', 'w')
    for domain in domains:
        print(domain)
        new_df = df[df['Domain']==domain]
        topics = new_df['topic_name'].unique()
        dialogue_scores, wizard_scores, seeker_scores = [],[],[]
        dialogue_dicts = defaultdict(list)
        for topic in topics:
            topic_df = new_df[new_df['topic_name']==topic]
            speakers = topic_df['speaker'].unique()
            scores = score_dist(topic_df, args.metric)
            wizard_scores.append(avg_tokens_wizard(topic_df, speakers, args.metric))
            #print(avg_tokens_wizard(topic_df, speakers, args.metric))
            seeker_scores.append(avg_tokens_information_seeker(topic_df, speakers, domain, args.metric))
            dialogue_scores.append(scores)
            dialogue_dicts[len(topic_df)].append(scores)
        print('Dialogue')
        print(dialogue_scores)
        print('--------------')
        print('Wizard')
        print(wizard_scores)
        print('--------------')
        print('Seeker')
        print(seeker_scores)
        print('--------------')
        dialogue_scores = [x for x in dialogue_scores if x==x]
        wizard_scores = [x for x in wizard_scores if x==x]
        seeker_scores = [x for x in seeker_scores if x==x]
        print(f"Domain:{domain}, Metric:{args.metric}, Average scores: {sum(dialogue_scores)/len(dialogue_scores)}")
        print(f"Domain:{domain}, Metric:{args.metric}, Average wizard scores: {sum(wizard_scores)/len(wizard_scores)}")
        print(f"Domain:{domain}, Metric:{args.metric}, Average seeker scores: {sum(seeker_scores)/len(seeker_scores)}\n")
        f.write(f"Domain:{domain}, Metric:{args.metric}, Average scores: {sum(dialogue_scores)/len(dialogue_scores)}\n")
        f.write(f"Domain:{domain}, Metric:{args.metric}, Average wizard scores: {sum(wizard_scores)/len(wizard_scores)}\n")
        f.write(f"Domain:{domain}, Metric:{args.metric}, Average seeker scores: {sum(seeker_scores)/len(seeker_scores)}\n")
        for item in dialogue_dicts:
            print(f"{item}\t{sum(dialogue_dicts[item])/len(dialogue_dicts[item])}")
            f.write(f"{item}\t{sum(dialogue_dicts[item])/len(dialogue_dicts[item])}\n")
        print("\n-----------------------------------------------------------------------------------------------\n")
        f.write("\n-----------------------------------------------------------------------------------------------\n")
    f.close()
                

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--metric", type=str, default="kprecision")
    parser.add_argument("--model", type=str, default="chatgpt")
    parser.add_argument("--output_path", type=str, default="chatgpt")
    args = parser.parse_args()
    main(args)





