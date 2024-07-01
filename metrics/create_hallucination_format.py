import os, ast, argparse
import numpy as np
import math



def get_responses(lines):
    lines = [x for x in lines if x!='\n']
    #lines = [line for line in lines if not 'Decision Maker' in line]
    responses = [x.split(":")[-1].strip() for x in lines]
    agents = [x.split(":")[0].strip() for x in lines]
    return responses, agents


def main(args):
    print(args.path)
    print(args.output_path)
    f = open(f"{args.output_path}/hallucination_outputs_{args.model}.txt", 'w')
    output_string ="Domain\ttopic_name\tspeaker\tknowledge\tresponse\n"
    f.write(output_string)

    for domain in ['debates', 'meta_reviews','helpful_reviews']:
        all_knowledge_files = os.listdir(f"{args.path}/{domain}/")
        for file in all_knowledge_files:
            knowledge_file = open(f"{args.path}/knowledge_prompts/{domain}/{file}")
            if os.path.exists(os.path.join(args.path, f"{domain}_{file}")):
                dialogue_file = open(os.path.join(args.path, f"{domain}_{file}"))
                knowledge = knowledge_file.readlines()
                if domain =='meta_review':
                    knowledge = [x for x in knowledge if x!='\n']
                    knowledge = ''.join(x.replace('\n','') for x in knowledge)
                else:
                    knowledge = knowledge[0].replace('\n','')
                dialogues = dialogue_file.readlines()

                responses, agents = get_responses(dialogues)
                #print(responses)

                for resp, agent in zip(responses, agents):
                    #print(resp)
                    #print(agent)
                    output_string=f"{domain}\t{file.strip('.txt')}\t{agent}\t{knowledge}\t{resp}\n"
                    f.write(output_string)
    
    f.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--model', help='Options are chatgpt, llama, mistral, mixtral', required=True)
    parser.add_argument('--path', help='Path to Dialogues', required=True)
    parser.add_argument('--output_path', help='Path to output', required=True)
    args = parser.parse_args()
    main(args)


