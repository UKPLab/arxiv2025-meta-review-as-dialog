import os
import statistics
import argparse

def main(args):
    prompts = ['prompt_1', 'prompt_2', 'prompt_3']
    for domain in ['helpful_reviews', 'debates', 'meta_reviews']:
        overall, wizard, seeker = [], [], []
        for prompt in prompts:
            out = os.path.join(args.out_path, prompt, f'model_level_metrics_{args.model}_{args.metric}.txt')
            if not os.path.exists(out):
                continue
            f = open(out, 'r')
            lines = f.readlines()
            all_lines =[]
            for line in lines:
                if domain in line:
                    all_lines.append(line)
            if len(all_lines) ==0:
                continue
            new_cont = [float(x.split(',')[-1].split(':')[-1].strip()) for x in all_lines]
            print(new_cont)
            print(len(new_cont))
            overall.append(new_cont[0])
            wizard.append(new_cont[1])
            seeker.append(new_cont[2])
        f = open(os.path.join(args.out_path, f'overall_{args.model}_{args.metric}_{domain}.txt'), 'w')
        overall_s = sum(overall)/len(overall)
        wizard_s = sum(wizard)/len(wizard)
        seeker_s = sum(seeker)/len(seeker)
        if len(overall) >1:
            f.write(f'overall: {overall_s}, STD:{statistics.stdev(overall)}\n')
            f.write(f'wizard: {wizard_s}, STD:{statistics.stdev(wizard)}\n')
            f.write(f'seeker: {seeker_s}, STD:{statistics.stdev(seeker)}\n')
        else:
            f.write(f'overall: {overall_s}\n, STD:0.0\n')
            f.write(f'wizard: {wizard_s}, STD:0.0\n')
            f.write(f'seeker: {seeker_s}, STD:0.0\n')
        
        
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='chatgpt')
    parser.add_argument('--metric', type=str, default='kprecision')
    parser.add_argument('--out_path', type=str)
    #parser.add_argument('--domain', type=str, default='helpful_reviews')
    args = parser.parse_args()
    main(args)


