from collections import Counter
import json
import os
import re
import string
import time
import evaluate
import openai

#Code adapted from https://github.com/McGill-NLP/instruct-qa.git

from tqdm import tqdm
from metric_class import Metric
#from instruct_qa.evaluation.metrics import BERTScore, F1
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelWithLMHead,
    AutoModelForQuestionAnswering, pipeline
)
import spacy
import pandas as pd
from allennlp.predictors.predictor import Predictor
import allennlp_models.pair_classification

#from instruct_qa.prompt.templates import HistoryTemplate, PromptTemplate

INVALID_QUESTION = -1
NO_ANS = "[CLS]"
NO_VALID_QUESTIONS = "NO_Q"
NO_Q = -1
ENTAILMENT_SCORE = 1
CONTRADICTION_SCORE = 0
NEUTRAL_SCORE = 0.5


class FaithDialCritic(Metric):
    """
    FaithDialCritic is a metric that measures the faithfulness of a response to a given evidence.
    0 - faithfull
    1 - unfaithfull
    lower score is better
    """

    def __init__(self, history_list, response_list, evidence_list, model, ids=None):
        self.history_list = history_list
        self.response_list = response_list
        self.evidence_list = evidence_list

        self.tokenizer = AutoTokenizer.from_pretrained(
                "McGill-NLP/roberta-large-faithcritic", return_tensors="pt"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
                "McGill-NLP/roberta-large-faithcritic",
        ).cuda()

    def compute_score(self):
        """
        history_list: list of list of strings (won't be used)
        response_list: list of strings
        evidence_list: list of list passages from collection - text, title, sub_title
        """
        evidence_list = self.evidence_list
        response_list = self.response_list

        scores = []
        print(f"Len of evidence list: {len(evidence_list)}")
        print(f"Len of response list: {len(response_list)}")
        for evidence_string, response in zip(evidence_list, response_list):
            print(evidence_string)
            print(response)
            #evidence_string = " ".join([e for e in evidence])
            input = self.tokenizer(
                evidence_string, response, return_tensors="pt", truncation=True
            )
            input = {key: val.cuda() for key, val in input.items()}
            output_logits = self.model(**input).logits
            score = torch.softmax(output_logits, dim=1)[:, 1].item()

            scores.append(score)

        return np.mean(scores), scores

    def compute_hallucination_scorer(self):
        pipe = pipeline("text-classification", model="McGill-NLP/roberta-large-faithcritic", truncation=True)
        evidence_list = self.evidence_list
        response_list = self.response_list
        all_data =[]
        for evidence_string, response in zip(evidence_list, response_list):
            string = evidence_string + "<s></s>" + response
            all_data.append(string)
        output = pipe(all_data)
        scores = [output[i]['score'] for i in range(len(output))]
        labels = [output[i]['label'] for i in range(len(output))]
        return scores, labels


class QSquared():
    # Code taken from https://github.com/McGill-NLP/instruct-qa/blob/main/instruct_qa/evaluation/faithfulness_metrics.py and https://github.com/orhonovich/q-squared
    def __init__(self, history_list, response_list, evidence_list, ids=None):
        self.history_list = history_list
        self.response_list = response_list
        self.evidence_list = evidence_list
        self.qg_tokenizer = AutoTokenizer.from_pretrained(
            "mrm8488/t5-base-finetuned-question-generation-ap"
        )
        self.qg_model = AutoModelWithLMHead.from_pretrained(
            "mrm8488/t5-base-finetuned-question-generation-ap"
        ).cuda()
        self.qa_tokenizer = AutoTokenizer.from_pretrained(
            "ktrapeznikov/albert-xlarge-v2-squad-v2"
        )
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
            "ktrapeznikov/albert-xlarge-v2-squad-v2",
        ).cuda()
        self.predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/snli_roberta-2020.06.09.tar.gz",
            predictor_name="textual_entailment",
        )

        self.nlp = spacy.load("en_core_web_sm")

    def compute_score(self):
        f1_scores = []
        nli_scores = []
        evidence_list = self.evidence_list
        response_list = self.response_list

        for evidence_string, response in zip(evidence_list, response_list):
            (
                f1_score,
                res_questions,
                res_cands,
                res_answers,
                res_scores,
            ) = self.get_response_score(
                response,
                evidence_string,
                gen_method="beam",
                single=True,
                remove_personal=True,
            )

            if f1_score == INVALID_QUESTION:
                res_questions = [NO_VALID_QUESTIONS]
                res_cands = [NO_VALID_QUESTIONS]
                res_answers = [NO_VALID_QUESTIONS]
                res_scores = [INVALID_QUESTION]

            f1_scores_instance = []
            nli_scores_instance = []
            for i in range(len(res_questions)):
                f1_score_q2 = res_scores[i]
                evidence_answer = str(res_answers[i])

                nli_score_q2 = f1_score_q2

                if (
                    0 <= f1_score_q2 < 1
                    and NO_ANS not in evidence_answer
                    and evidence_answer != ""
                    and evidence_answer != "nan"
                ):
                    f1_scores_instance.append(f1_score_q2)

                    nli_label = self.get_nli_label(
                        str(res_questions[i]), str(res_cands[i]), evidence_answer
                    )

                    if nli_label == "entailment":
                        nli_score_q2 = ENTAILMENT_SCORE
                    elif nli_label == "contradiction":
                        nli_score_q2 = CONTRADICTION_SCORE

                elif f1_score_q2 == NO_Q:
                    nli_fallback = self.get_e2e_nli_score(
                        str(response), str(evidence_string).lower()
                    )
                    nli_score_q2 = nli_fallback
                    f1_scores_instance.append(nli_fallback)
                else:
                    f1_scores_instance.append(f1_score_q2)

                nli_scores_instance.append(nli_score_q2)

            f1_scores.append(np.mean(f1_scores_instance))
            nli_scores.append(np.mean(nli_scores_instance))

            all_scores = [
                    {"f1": f1_score, "nli": nli_score}
                    for f1_score, nli_score in zip(f1_scores, nli_scores)
                ]
        return np.mean(f1_scores),np.mean(nli_scores), all_scores

    def get_answer(
        self, question, text
    ):  # Code taken from https://huggingface.co/transformers/task_summary.html
        inputs = self.qa_tokenizer.encode_plus(
            question,
            text,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
        )
        inputs = {key: val.cuda() for key, val in inputs.items()}
        input_ids = inputs["input_ids"].tolist()[0]

        text_tokens = self.qa_tokenizer.convert_ids_to_tokens(input_ids)
        answer_start_scores, answer_end_scores = self.qa_model(
            **inputs, return_dict=False
        )

        answer_start = torch.argmax(
            answer_start_scores
        )  # Get the most likely beginning of answer with the argmax of the score
        answer_end = (
            torch.argmax(answer_end_scores) + 1
        )  # Get the most likely end of answer with the argmax of the score

        ans = self.qa_tokenizer.convert_tokens_to_string(
            self.qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
        )
        return ans

    def get_answer_candidates(self, text):
        doc = self.nlp(text)
        candidates = [ent.text for ent in list(doc.ents)]
        noun_chunks = list(doc.noun_chunks)
        for chunk in noun_chunks:
            found = False
            for cand in candidates:
                if chunk.text.lower() == cand.lower():
                    found = True
            if not found:
                candidates.append(chunk.text)
        candidates += [chunk.text for chunk in list(doc.noun_chunks) if chunk.text not in candidates]
        candidates = [cand for cand in candidates if cand.lower() != "i"]
        return candidates

    def get_question_greedy(self, answer, context, max_length=128):
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = self.qg_tokenizer([input_text], return_tensors="pt", truncation=True)

        output = self.qg_model.generate(
            input_ids=features["input_ids"].cuda(),
            attention_mask=features["attention_mask"].cuda(),
            max_length=max_length,
        )

        question = self.qg_tokenizer.decode(output[0]).replace("question: ", "", 1)
        return question

    def get_questions_beam(
        self, answer, context, max_length=128, beam_size=5, num_return=5
    ):
        all_questions = []
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = self.qg_tokenizer([input_text], return_tensors="pt", truncation=True)

        beam_outputs = self.qg_model.generate(
            input_ids=features["input_ids"].cuda(),
            attention_mask=features["attention_mask"].cuda(),
            max_length=max_length,
            num_beams=beam_size,
            no_repeat_ngram_size=3,
            num_return_sequences=num_return,
            early_stopping=True,
        )

        for beam_output in beam_outputs:
            all_questions.append(
                self.qg_tokenizer.decode(beam_output, skip_special_tokens=True).replace(
                    "question: ", "", 1
                )
            )
        print(all_questions)
        return all_questions

    def get_questions_sample(
        self, answer, context, max_length=128, top_k=50, top_p=0.95, num_return=5
    ):
        all_questions = []
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = self.qg_tokenizer([input_text], return_tensors="pt", truncation=True)

        sampled_outputs = self.qg_model.generate(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            max_length=max_length,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return,
        )

        for sampled in sampled_outputs:
            all_questions.append(
                self.qg_tokenizer.decode(sampled, skip_special_tokens=True).replace(
                    "question: ", "", 1
                )
            )
        print(all_questions)

        return all_questions

    def non_personal(self, question):
        question_tok = self.nlp(question)
        for tok in question_tok:
            if tok.dep_ == "nsubj":
                if tok.text.lower() == "i" or tok.text.lower() == "you":
                    return False
            elif tok.dep_ == "poss":
                if tok.text.lower() == "my" or tok.text.lower() == "your":
                    return False
        return True

    def clean_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\b(a|an|the|in|our)\b", " ", text)
        return re.sub(" +", " ", text).strip()

    def filter_questions(self, exp_ans, pred_ans):
        if pred_ans == NO_ANS:
            return "NO MATCH"
        if self.clean_text(exp_ans) != self.clean_text(pred_ans):
            return "NO MATCH"
        return "VALID"

    def f1_score(self, a_gold, a_pred):
        if a_pred == "":
            return 0
        gold_toks = self.clean_text(a_gold).split()
        pred_toks = self.clean_text(a_pred).split()
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def single_question_score(self, question, cand, response, knowledge):
        pred_ans = self.get_answer(question, response)

        if self.filter_questions(cand, pred_ans) == "VALID":
            knowledge_ans = self.get_answer(question, knowledge)
            if knowledge_ans != NO_ANS:
                return self.f1_score(cand, knowledge_ans), knowledge_ans
            else:
                return 0, NO_ANS
        else:
            return INVALID_QUESTION, INVALID_QUESTION

    def get_response_score(
        self, response, knowledge, gen_method, single, remove_personal=True
    ):
        f1 = 0
        num_questions = 0

        valid_questions = []
        valid_cands = []
        knowledge_answers = []
        scores = []

        candidates = self.get_answer_candidates(response)
        for cand in candidates:
            if gen_method == "greedy":
                questions = [self.get_question_greedy(cand, response)]
            elif gen_method == "beam":
                questions = self.get_questions_beam(cand, response)
            else:
                questions = self.get_questions_sample(cand, response)

            for question in questions:
                if not remove_personal or self.non_personal(question):
                    question_score, knowledge_ans = self.single_question_score(
                        question, cand, response, knowledge
                    )
                    if question_score != INVALID_QUESTION:
                        num_questions += 1
                        f1 += question_score

                        valid_questions.append(question)
                        valid_cands.append(cand)
                        knowledge_answers.append(knowledge_ans)
                        scores.append(question_score)

                        if single:
                            break
        if num_questions:
            avg_f1 = f1 / num_questions
        else:
            avg_f1 = INVALID_QUESTION
        return avg_f1, valid_questions, valid_cands, knowledge_answers, scores

    def get_e2e_nli_score(self, response, knowledge):
        res = self.predictor.predict(premise=knowledge, hypothesis=response)

        nli_label = res["label"]

        if nli_label == "entailment":  # If entails, the score is 1
            return ENTAILMENT_SCORE
        elif nli_label == "contradiction":  # If contradicts, the score is 0
            return CONTRADICTION_SCORE
        else:
            return NEUTRAL_SCORE

    def get_nli_label(self, question, cand, evidence_ans):
        premise = question + " " + evidence_ans + "."
        hypothesis = question + " " + cand + "."

        res = self.predictor.predict(premise=premise, hypothesis=hypothesis)

        return res["label"]

class KBERTScore():

    def __init__(self, history_list, response_list, evidence_list, ids=None):
        #evidence_strings = [
        #    " ".join([e for e in evidence]) for evidence in evidence_list
        #]
        self.history_list = history_list
        self.response_list = response_list
        self.evidence_list = evidence_list

        self._metric = evaluate.load("bertscore", rescale_with_baseline=True)
        self.evidence_strings = evidence_list
        
    def compute_scores(self):
        scores = self._metric.compute(
            predictions=self.response_list, references=self.evidence_strings, lang="en")

        individual_scores = []
        for i in range(len(self.response_list)):
            individual_scores.append(
                {
                    "precision": scores["precision"][i],
                    "recall": scores["recall"][i],
                    "f1": scores["f1"][i],
                }
            )
            #self.save_individual_scores(ids, individual_scores)
        print(np.mean(scores["precision"]), np.mean(scores["recall"]), np.mean(scores["f1"]), individual_scores)
        return np.mean(scores["precision"]), np.mean(scores["recall"]), np.mean(scores["f1"]), individual_scores


class KPrecision():
    def __init__(self, history_list, response_list, evidence_list, ids=None):
        #evidence_strings = [
        #    [" ".join([e for e in evidence])] for evidence in evidence_list
        #]
        self.evidence_list = evidence_list
        self.response_list = response_list
        self.history_list = history_list

    def compute_score(self):
        evidence_strings = self.evidence_list
        #print(evidence_strings[0])
        #print(self.response_list[0])
        
        self.scores = [
            self._precision(prediction, reference)
            for prediction, reference in zip(self.response_list, evidence_strings)
        ]
        print([{"kprecision": score} for score in self.scores])

        return np.mean(self.scores), self.scores
    

    def _precision(self, prediction, references):
        precision_scores = [
            self._precision_score(prediction, references)
        ]
        return max(precision_scores)
    

    def _normalize_text(self,text):
        if isinstance(text, str):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            text = text.lower()
            text = "".join(char for char in text if char not in set(string.punctuation))
            text = re.sub(regex, " ", text)
            text = " ".join(text.split())
        return text
    
    def _get_tokens(self, text):
        if not isinstance(text,str):
            return self._normalize_text(str(text))
        return self._normalize_text(text).split()
    
    def _precision_score(self, prediction, reference):

        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)
        print(reference_tokens)
        print(prediction_tokens)

        common_tokens = Counter(reference_tokens) & Counter(prediction_tokens)

        num_common = sum(common_tokens.values())

        if len(prediction_tokens) == 0:
            # if prediction is empty, precision is 0
            return 0

        if num_common == 0:
            return 0

        precision = 1.0 * num_common / len(prediction_tokens)

        return precision
