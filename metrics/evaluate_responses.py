from metrics import QSquared, FaithDialCritic, KBERTScore, KPrecision
import pandas as pd
import argparse



def kprecision(df, history_list, responses, knowledge, model, output_path):
        #faithdial_scores = FaithDialCritic(responses, knowledge)
        kprecision_mean, all_scores = KPrecision(history_list, responses, knowledge).compute_score()
        #kprecision_mean, all_scores = kprecision.compute_score()

        print(kprecision_mean)
        print(len(df))
        print(len(all_scores))
        df['kprecision'] = all_scores
        df.to_csv(f'{output_path}/hallucination_{model}_outputs_kprecision.txt', sep='\t')

def kbert(df, history_list, responses, knowledge, model, output_path):
    p_mean,r_mean,f1_mean, indiv_scores = KBERTScore(history_list, responses, knowledge).compute_scores()
    p_sc, r_sc, f1_sc = [],[],[]
    print(indiv_scores)
    for idx in range(len(indiv_scores)):
        p_sc.append(indiv_scores[idx]["precision"])
        r_sc.append(indiv_scores[idx]["recall"])
        f1_sc.append(indiv_scores[idx]["f1"])
    df['prcision'], df['recall'], df['f1']= p_sc, r_sc, f1_sc
    df.to_csv(f'{output_path}/hallucination_{model}_outputs_kbert.txt', sep='\t')

def q_squared(df, history_list, responses, knowledge, model, output_path):
    f1_mean, nli_mean, all = QSquared(history_list, responses, knowledge).compute_score()
    f1,nli = [], []
    for idx in range(len(all)):
        f1.append(all[idx]["f1"])
        nli.append(all[idx]["nli"])
    df['f1'], df['nli']= f1, nli
    df.to_csv(f'{output_path}/hallucination_{model}_outputs_q2.txt', sep='\t')
    
def faith_critic(df, history_list, responses, knowledge, model, output_path):
    faith_mean, faith = FaithDialCritic(history_list, responses, knowledge,model).compute_score()
    df['faithdial_scores'] = faith
    df.to_csv(f'{output_path}/hallucination_{model}_outputs_faith_dial_critic.txt', sep='\t')

def faith_critic_labels(df, history_list, responses, knowledge, model, output_path):
    labels, scores = FaithDialCritic(history_list, responses, knowledge, model).compute_hallucination_scorer()
    df['labels'] = labels
    df['scores'] = scores
    df.to_csv(f'{output_path}/hallucination_{model}_outputs_faith_dial_critic_labels.txt', sep='\t')    

metric_map = {"faithcritic":faith_critic, "kbert":kbert, "kprecision":kprecision, "q2":q_squared, "faithcritic_labels":faith_critic_labels}

def main(args):
    df = pd.read_csv(args.path, sep ='\t', on_bad_lines='skip')
    df = df.dropna()
    responses = df['response']
    knowledge = df['knowledge']
    #print(f"Len of responses: {len(responses)}")
    #print(f"Len of knowledge: {len(knowledge)}")
    #print(responses)
    #print(knowledge)
    history_list =[]
    ids = [i for i in range(0, len(responses))]
    metric = metric_map[args.metric]
    metric(df, history_list, responses, knowledge, args.model, args.output_path)

if __name__=='__main__':
      parser = argparse.ArgumentParser(description='args')
      parser.add_argument('--metric', help='Options are faithcritic, q2, kbert, kprecision, faithcritic_labels', required=True)
      parser.add_argument('--path', help='Path to Outputs', required=True)
      parser.add_argument('--output_path', help='Path to Output scores', required=True)
      parser.add_argument('--model', help='Options are chatgpt, llama, dagstuhl (just the data)', required=True)
      args = parser.parse_args()
      main(args)