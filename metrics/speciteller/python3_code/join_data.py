import pandas as pd
import argparse, os

def find_com(df_know, df_spec):
    cllect_resp=[]
    for resp in df_spec['response'].tolist():
        if resp not in df_know['response'].tolist():
            print(resp)
            cllect_resp.append(resp)
            print('\n')
    df_spec=df_spec[~df_spec['response'].isin(cllect_resp)]
    return df_know, df_spec

def unbalanced_data(df_know, df_spec):
    #df = pd.read_csv('outputs/prompt_3/trial.txt', sep='\t')
    df_know = df_know.dropna()
    df_know.drop_duplicates(keep=False, inplace=True)
    df_spec = df_spec.dropna()
    df_spec.drop_duplicates(keep=False, inplace=True)
    common_response = set(df_know['response'].tolist() and df_spec['response'].tolist())
    df_know = df_know[df_know['response'].isin(common_response)]
    df_spec = df_spec[df_spec['response'].isin(common_response)]
    df_know, df_spec  = find_com(df_know, df_spec)
    return df_know, df_spec

def main(args):
    df = pd.read_csv(args.data_path, sep='\t', on_bad_lines='skip')
    print(f'Len of df: {len(df)}')
    specifity = pd.read_csv(args.specifity_path, sep='\t', names =['response', 'scores'], on_bad_lines='skip')
    print(f'Len of spec df:{len(specifity)}')
    if len(df)!=len(specifity):
        df, specifity = unbalanced_data(df, specifity)
    scores = specifity['scores'].to_list()
    df['specificty'] = scores
    df.to_csv(f'{args.output_path}/hallucination_{args.model}_outputs_specifity.txt', sep='\t', index=False)    


if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, default='/storage/ukp/work/purkayastha/amazon_products/chatgpt_dialogues/hallucination_outputs_chatgpt.txt')
    argparser.add_argument('--specifity_path', type=str, default='/storage/ukp/work/purkayastha/amazon_products/chatgpt_dialogues/specifity_output_chatgpt.txt')
    argparser.add_argument('--model', type=str, default='chatgpt')
    argparser.add_argument('--output_path', type=str, default='chatgpt')
    args = argparser.parse_args()
    main(args)