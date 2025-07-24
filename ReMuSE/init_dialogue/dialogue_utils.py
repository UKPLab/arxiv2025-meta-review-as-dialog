import pandas as pd

def helpful_review_dialogues(data_path, prompts_for_helpful_reviews):
    help_df = pd.read_csv(f'{data_path}/helpful_reviews.tsv', sep='\t')
    prod_prompts, prod_names,ids = [],[],[]

    for idx, rows in help_df.iterrows():
        prod_prompts.append(prompts_for_helpful_reviews.format(prod = rows["titles"], reviews = rows["reviews"]))
        prod_names.append(rows["titles"])
        ids.append(rows["id"])

    return prod_prompts, prod_names, ids



def debate_dialogues(data_path, prompts_for_debates):


    debate_df = pd.read_csv(f'{data_path}/debates.tsv', sep='\t')
    debate_prompts, debate_names, ids=[], [],[]
        
    for idx, rows in debate_df.iterrows():
        debate_prompts.append(prompts_for_debates.format(topic = rows["topic"], for_args = rows["for_args"], against_args = rows["against_args"]))
        debate_names.append(rows["topic"])
        ids.append(rows["ID"])
    return debate_prompts, debate_names, ids




def meta_review_dialogues(data_path, prompts_for_meta_reviews):
    mr_df = pd.read_csv(f'{data_path}/meta_reviews.tsv', sep='\t')

    mr_prompts, paper_names, ids = [],[],[]

    for idx, rows in mr_df.iterrows():
        mr_prompts.append(prompts_for_meta_reviews.format(paper = rows["Paper"], type = rows["Type"], review1 = rows["Review 1"], review2 = rows["Review 2"], review3 = rows["Review 3"]))
        paper_names.append(rows["Paper"])
        ids.append(rows["ID"])
    
    return mr_prompts, paper_names, ids
        
