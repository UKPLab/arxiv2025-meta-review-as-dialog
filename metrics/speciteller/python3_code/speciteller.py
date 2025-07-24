
# coding: utf-8

import argparse
import glob
import liblinear.liblinearutil as ll
#rebuttal_list = ['rebuttal_followup', 'rebuttal_concede-criticism', 'rebuttal_by-cr', 'rebuttal_future', 'rebuttal_done']
aspects = ['arg_other','asp_clarity','asp_meaningful-comparison','asp_motivation-impact','asp_originality','asp_replicability','asp_soundness-correctness','asp_substance']
all_rebuttals=['rebuttal_accept-praise','rebuttal_answer', 'rebuttal_by-cr', 'rebuttal_concede-criticism', 'rebuttal_contradict-assertion','rebuttal_done','rebuttal_followup', 'rebuttal_future', 'rebuttal_mitigate-criticism', 'rebuttal_other', 'rebuttal_refute-question', 'rebuttal_reject-criticism', 'rebuttal_social', 'rebuttal_structuring', 'rebuttal_summary', 'rebuttal_reject-request']
import utils
import sys
import os
import pandas as pd
from features import Space
from generatefeatures import ModelNewText

RT = "/storage/ukp/work/purkayastha/buy_this_paper/src/metrics/speciteller"

BRNCLSTSPACEFILE = os.path.join(RT,"cotraining_models/brnclst1gram.space")
SHALLOWSCALEFILE = os.path.join(RT,"cotraining_models/shallow.scale")
SHALLOWMODELFILE = os.path.join(RT,"cotraining_models/shallow.model")
NEURALBRNSCALEFILE = os.path.join(RT,"cotraining_models/neuralbrn.scale")
NEURALBRNMODELFILE = os.path.join(RT,"cotraining_models/neuralbrn.model")

def initBrnSpace():
    s = Space(101)
    s.loadFromFile(BRNCLSTSPACEFILE)
    return s

def readScales(scalefile):
    scales = {}
    with open(scalefile) as f:
        for line in f:
            k,v = line.strip().split("\t")
            scales[int(k)] = float(v)
        f.close()
    return scales

brnclst = utils.readMetaOptimizeBrownCluster()
embeddings = utils.readMetaOptimizeEmbeddings()
brnspace = initBrnSpace()
scales_shallow = readScales(SHALLOWSCALEFILE)
scales_neuralbrn = readScales(NEURALBRNSCALEFILE)
model_shallow = ll.load_model(SHALLOWMODELFILE)
model_neuralbrn = ll.load_model(NEURALBRNMODELFILE)

def simpleScale(x, trainmaxes=None):
    maxes = trainmaxes if trainmaxes!=None else {}
    if trainmaxes == None:
        for itemd in x:
            for k,v in itemd.items():
                if k not in maxes or maxes[k] < abs(v): maxes[k] = abs(v)
    newx = []
    for itemd in x:
        newd = dict.fromkeys(itemd)
        for k,v in itemd.items():
            if k in maxes and maxes[k] != 0: newd[k] = (v+0.0)/maxes[k]
            else: newd[k] = 0.0
        newx.append(newd)
    return newx,maxes

def getFeatures(fin):
    aligner = ModelNewText(brnspace,brnclst,embeddings)
    aligner.loadFromFile(fin)
    aligner.fShallow()
    aligner.fNeuralVec()
    aligner.fBrownCluster()
    y,xs = aligner.transformShallow()
    _,xw = aligner.transformWordRep()
    return y,xs,xw

def score(p_label, p_val):
    ret = []
    for l,prob in zip(p_label,p_val):
        m = max(prob)
        if l == 1: ret.append(1-m)
        else: ret.append(m)
    return ret

def predict(y,xs,xw):
    xs,_ = simpleScale(xs,scales_shallow)
    xw,_ = simpleScale(xw,scales_neuralbrn)
    p_label, p_acc, p_val = ll.predict(y,xs,model_shallow,'-q -b 1')
    ls_s = score(p_label,p_val)
    p_label, p_acc, p_val = ll.predict(y,xw,model_neuralbrn,'-q -b 1')
    ls_w = score(p_label,p_val)
    return [(x+y)/2 for x,y in zip(ls_s,ls_w)],ls_s,ls_w

def find_com(response, spec_lines, spec_preds):
    idxs =[]
    for i in range(len(spec_lines)):
        if response.strip('\n') == spec_lines[i]:
            idxs.append(i)
    preds_final=[spec_preds[i] for i in idxs]
    return preds_final


def writeSpecificity(preds, lines, outf, response_df, input_path, model):
    with open(outf,'w') as f:
        for t,x in zip(lines,preds):
            t= t.strip('\n')
            #f.write("%f\n" % x)
            f.write(t+'\t'+str(x)+'\n')
        f.close()

    #lines=[x.strip('\n') for x in lines]
    #preds_final = find_com(lines, response_df['response'], preds)
    #response_df['specificty'] = preds_final
    #response_df.to_csv(f'{input_path}/hallucination_{model}_outputs_specifity.txt', sep='\t', index=False)
    #print("Output to "+outf+" done.")

def run(identifier, sentlist):
    ## main function to run speciteller and return predictions
    ## sentlist should be a list of sentence strings, tokenized;
    ## identifier is a string serving as the header of this sentlst
    aligner = ModelNewText(brnspace,brnclst,embeddings)
    aligner.loadSentences(identifier, sentlist)
    aligner.fShallow()
    aligner.fNeuralVec()
    aligner.fBrownCluster()
    y,xs = aligner.transformShallow()
    _,xw = aligner.transformWordRep()
    preds_comb, preds_s, preds_w = predict(y,xs,xw)
    return preds_comb

def strip_filename(filename):
    return filename.split('/')[-1].strip('.txt')

def specificity_finder(filename, path, model, response_df, input_path):
    f = open(filename)
    lines = f.readlines()
    y,xs,xw = getFeatures(filename)
    preds_comb, preds_s, preds_w = predict(y,xs,xw)
    outputfile = os.path.join(path,f'specifity_output_{model}.txt')
    print('Len of prediction')
    print(len(preds_comb))
    writeSpecificity(preds_comb,lines, outputfile, response_df, input_path, model)

def main(args):
    input_path = args.path
    response_df = pd.read_csv(os.path.join(input_path,args.filename), sep='\t', on_bad_lines='skip')
    print('Response df')
    print(len(response_df))
    response_df.to_csv(f'{input_path}/trial.txt', sep='\t', index=False)
    responses = response_df['response']
    filename = os.path.join(input_path, f'specifity_input_{args.model}.txt')
    f = open(filename , 'w')
    for resp in responses:
        f.write(str(resp)+'\n')
    f.close()
    specificity_finder(filename, input_path, args.model, response_df, input_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path", help="path of the file", required=True)
    argparser.add_argument("--filename", help="filename for the file", required=True)
    argparser.add_argument("--model", help="model choices: chatgpt, llama", required=True)
    #argparser.add_argument("--outputfile", help="output file to save the specificity scores", required=True)
    #argparser.add_argument("--write_all_preds", help="write predictions from individual models in addition to the overall one", action="store_true")
    #argparser.add_argument("--tokenize", help="tokenize input sentences?", required=True)
    sys.stderr.write("SPECITELLER: please make sure that your input sentences are WORD-TOKENIZED for better prediction.\n")
    args = argparser.parse_args()
    main(args)    

            
                

