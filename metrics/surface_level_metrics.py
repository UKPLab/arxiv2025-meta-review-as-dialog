import pandas as pd
import argparse, os
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter


domains =['debates', 'helpful_reviews', 'meta_reviews']

seeker_map = {"meta_reviews": "meta reviewer", "helpful_reviews": "buyer", "debates": "decision maker"}
wizard_name = ["dialogue agent"]

def preprocessing(x):
    return x.lower().replace("-"," ").replace('"','')

def count_tokens(df, col_name):
    tokens = df[col_name].apply(lambda x: len(x.split()))
    return tokens

def turn_logic(lines):
    current_speaker = ""
    turns = 0
    for line in lines:
        parts = line.split(":")
        if len(parts) > 1:
            speaker = parts[0]
            text = parts[1]
            if speaker != current_speaker:
                turns +=1 
                current_speaker = speaker
    return turns
    
def no_of_turns(path, domain):
    all_files = os.listdir(path)
    all_turn_counts = []
    for files in all_files:
        fn = files.strip('.txt').replace("/","_").replace('\t','')
        if domain in fn:
            f = open(os.path.join(path,files))
            lines = f.readlines()
            turns = turn_logic(lines)
            all_turn_counts.append(turns)
    print(all)
    mean = sum(all_turn_counts)/len(all_turn_counts)
    min_turns = min(all_turn_counts)
    max_turns = max(all_turn_counts)
    return mean, min_turns, max_turns

def avg_tokens_wizard(df, wizard_names):

    wizard_names =[x for x in wizard_names if preprocessing(x) in wizard_name]
    print('Filtered wizard names')
    print(wizard_names)
    wizard = df[df['speaker'].isin(wizard_names)]
    bigram, trigram, fourgram = diversity_score(wizard, 'response')

    wizard_tokens = count_tokens(wizard, 'response')

    return wizard_tokens.mean(), min(wizard_tokens), max(wizard_tokens), bigram, trigram, fourgram

def avg_tokens_information_seeker(df, seeker_names, domain):
    seeker_names =[x for x in seeker_names if preprocessing(x) == seeker_map[domain]]
    print('Filtered seeker names')
    print(seeker_names)

    seeker = df[df['speaker'].isin(seeker_names)]
    seeker_tokens = count_tokens(seeker, 'response')
    bigram, trigram, fourgram = diversity_score(seeker, 'response')

    return seeker_tokens.mean(), min(seeker_tokens), max(seeker_tokens), bigram, trigram, fourgram

def avg_tokens_knowledge(df):
    knowledge_tokens = count_tokens(df, 'knowledge')

    return knowledge_tokens.mean(), min(knowledge_tokens), max(knowledge_tokens)

def diversity_score(df, col_name):
    data = df[col_name].tolist()
    all_data = '. '.join(x for x in data)
    tokens = nltk.word_tokenize(all_data)
    bigrams = ngrams(tokens,2)
    trigrams = ngrams(tokens,3)
    fourgrams = ngrams(tokens,4)
    
    return Counter(bigrams), Counter(trigrams), Counter(fourgrams)


def main(args):
    df = pd.read_csv(args.data_path, sep='\t', on_bad_lines='skip')
    df = df.dropna()
    df.columns = df.columns.str.replace(' ', '')

    f = open(os.path.join(args.output_path, f'surface_level_metrics_{args.model}.txt'), 'w')
    for domain in domains:
        print(f"Domain: {domain}")
        new_df = df[df['Domain'] == domain]
        speaker_names = set(new_df['speaker'].tolist())
        #speaker_names = [x.replace('-',' ') for x in speaker_names]
        speaker_names = [x for x in speaker_names if len(x.split()) <3]
        print(speaker_names)

        mean_wizard, minimum_wizard, maximum_wizard, bg_wz, tg_wz, fg_wz = avg_tokens_wizard(new_df, speaker_names)
        mean_seeker, minimum_seeker, maximum_seeker, bg_is, tg_is, fg_is = avg_tokens_information_seeker(new_df, speaker_names, domain)
        f.write(f"Domain: {domain}, Minimum number of Wizard tokens: {minimum_wizard}, Maximum number of Wizard tokens: {maximum_wizard}, Average number of Wizard tokens: {mean_wizard}\n")
        f.write(f"Domain: {domain}, Minimum number of Seeker tokens: {minimum_seeker}, Maximum number of Seeker tokens: {maximum_seeker}, Average number of Seeker tokens: {mean_seeker}\n")
        
        print(f"Domain: {domain}, Minimum number of Wizard tokens: {minimum_wizard}, Maximum number of Wizard tokens: {maximum_wizard}, Average number of Wizard tokens: {mean_wizard}")
        print(f"Domain: {domain}, Minimum number of Seeker tokens: {minimum_seeker}, Maximum number of Seeker tokens: {maximum_seeker}, Average number of Seeker tokens: {mean_seeker}")
        mean_knowledge, minimum_knowledge, maximum_knowledge = avg_tokens_knowledge(new_df)
        f.write(f"Domain: {domain}, Minimum number of Knowledge tokens: {minimum_knowledge}, Maximum number of Knowledge tokens: {maximum_knowledge}, Average number of Knowledge tokens: {mean_knowledge}\n")
        print(f"Domain: {domain}, Minimum number of Knowledge tokens: {minimum_knowledge}, Maximum number of Knowledge tokens: {maximum_knowledge}, Average number of Knowledge tokens: {mean_knowledge}")
        mean_turns, min_turns, max_turns = no_of_turns(args.data_folder, domain)
        print(f"Domain: {domain}, Minimum number of turns: {min_turns}, Maximum number of turns: {max_turns}, Average number of turns: {mean_turns}")
        print("--------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------")
        f.write(f"Domain: {domain}, Minimum number of turns: {min_turns}, Maximum number of turns: {max_turns}, Average number of turns: {mean_turns}\n")
        print(f"Domain: {domain}, Wizard bigrams: {len(bg_wz.keys())}, Wizard trigrams: {len(tg_wz.keys())}, Wizard fourgrams: {len(fg_wz.keys())}\n")
        f.write(f"Domain: {domain}, Wizard bigrams: {len(bg_wz.keys())}, Wizard trigrams: {len(tg_wz.keys())}, Wizard fourgrams: {len(fg_wz.keys())}\n")
        print(f"Domain: {domain}, Seeker bigrams: {len(bg_is.keys())}, Seeker trigrams: {len(tg_is.keys())}, Seeker fourgrams: {len(fg_is.keys())}\n")
        f.write(f"Domain: {domain}, Seeker bigrams: {len(bg_is.keys())}, Seeker trigrams: {len(tg_is.keys())}, Seeker fourgrams: {len(fg_is.keys())}\n")
    f.close()



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str)    
    argparser.add_argument('--data_folder', type=str)
    argparser.add_argument('--output_path', type=str)
    argparser.add_argument('--model', type=str, default='chatgpt')
    args = argparser.parse_args()
    main(args)
