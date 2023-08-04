import pandas as pd
import json
import pickle
import numpy as np
from utils import ROOT_DIR
from datasets import load_dataset as load_dataset_huggingface
from essential_generators import DocumentGenerator
import csv

def get_texts_and_labels(dataset, key_text='text', key_label='label', labels_of_interests=None):
    texts = []
    labels = []
    label_names = {}
    for entry in dataset:
        text = entry[key_text]
        if len(text) <= 1:
            continue
        else:
            label = int(entry[key_label])
            if labels_of_interests is not None:
                if label in labels_of_interests:
                    texts.append(text)
                    labels.append(label)
            else:
                texts.append(text)
                labels.append(label)
    return texts, labels

def read_csv_lmbff(csv_file):
    labels = []
    texts = []
    with open(csv_file, encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            label, text = row
            labels.append(int(label))
            texts.append(text)
    return texts, labels

def load_sst5():
    dataset_train = load_dataset_huggingface("SetFit/sst5", split='train')
    dataset_test = load_dataset_huggingface("SetFit/sst5", split='test')
    train_texts, train_labels= get_texts_and_labels(dataset_train, 'text', 'label')
    test_texts, test_labels= get_texts_and_labels(dataset_test, 'text', 'label')
    return train_texts, train_labels, test_texts, test_labels

def load_financial_phrasebank():
    dataset_train = load_dataset_huggingface("financial_phrasebank", 'sentences_75agree', split='train')
    dataset_test = load_dataset_huggingface("financial_phrasebank", 'sentences_75agree', split='train') # no test set
    train_texts, train_labels = get_texts_and_labels(dataset_train, 'sentence', 'label')
    test_texts, test_labels = get_texts_and_labels(dataset_test, 'sentence', 'label')
    return train_texts, train_labels, test_texts, test_labels

def load_poem_sentiment():
    dataset_train = load_dataset_huggingface("poem_sentiment", split='train')
    dataset_test = load_dataset_huggingface("poem_sentiment", split='train')    # test and validation set too small
    train_texts, train_labels = get_texts_and_labels(dataset_train, 'verse_text', 'label')
    test_texts, test_labels = get_texts_and_labels(dataset_test, 'verse_text', 'label')
    return train_texts, train_labels, test_texts, test_labels

def load_hate_speech18():
    dataset_train = load_dataset_huggingface("hate_speech18", split='train')
    dataset_test = load_dataset_huggingface("hate_speech18", split='train')    # no test set
    train_texts, train_labels = get_texts_and_labels(dataset_train, 'text', 'label', labels_of_interests=[0,1])
    test_texts, test_labels = get_texts_and_labels(dataset_test, 'text', 'label', labels_of_interests=[0,1])
    return train_texts, train_labels, test_texts, test_labels

def load_ethos_binary():
    dataset_train = load_dataset_huggingface("ethos", 'binary', split='train')
    dataset_test = load_dataset_huggingface("ethos", 'binary', split='train')    # no test set
    train_texts, train_labels = get_texts_and_labels(dataset_train, 'text', 'label')
    test_texts, test_labels = get_texts_and_labels(dataset_test, 'text', 'label')
    return train_texts, train_labels, test_texts, test_labels

def load_ethos_national_origin():
    dataset_train = load_dataset_huggingface("ethos", 'multilabel', split='train')
    dataset_test = load_dataset_huggingface("ethos", 'multilabel', split='train')    # no test set
    train_texts, train_labels = get_texts_and_labels(dataset_train, 'text', 'national_origin')
    test_texts, test_labels = get_texts_and_labels(dataset_test, 'text', 'national_origin')
    return train_texts, train_labels, test_texts, test_labels

def load_ethos_race():
    dataset_train = load_dataset_huggingface("ethos", 'multilabel', split='train')
    dataset_test = load_dataset_huggingface("ethos", 'multilabel', split='train')    # no test set
    train_texts, train_labels = get_texts_and_labels(dataset_train, 'text', 'race')
    test_texts, test_labels = get_texts_and_labels(dataset_test, 'text', 'race')
    return train_texts, train_labels, test_texts, test_labels

def load_ethos_religion():
    dataset_train = load_dataset_huggingface("ethos", 'multilabel', split='train')
    dataset_test = load_dataset_huggingface("ethos", 'multilabel', split='train')    # no test set
    train_texts, train_labels = get_texts_and_labels(dataset_train, 'text', 'religion')
    test_texts, test_labels = get_texts_and_labels(dataset_test, 'text', 'religion')
    return train_texts, train_labels, test_texts, test_labels

def load_medical_questions_pairs(return_pair=False):
    dataset_train = load_dataset_huggingface("medical_questions_pairs", split='train')
    dataset_test = load_dataset_huggingface("medical_questions_pairs", split='train')   # no test set
    def process(dataset):
        questions = []
        labels = []
        for entry in dataset:
            q_1 = entry['question_1']
            q_2 = entry['question_2']
            if return_pair:
                questions.append([q_1, q_2])
            else:
                questions.append(f"Question 1: {q_1} \n Question 2: {q_2}")
            labels.append(int(entry['label']))
        return questions, labels
    train_texts, train_labels = process(dataset_train)
    test_texts, test_labels = process(dataset_test)
    return train_texts, train_labels, test_texts, test_labels

def load_mrpc(return_pair=False):
    dataset_train = load_dataset_huggingface("SetFit/mrpc", split='train')
    dataset_test = load_dataset_huggingface("SetFit/mrpc", split='validation')
    def process(dataset):
        questions = []
        labels = []
        for entry in dataset:
            q_1 = entry['text1']
            q_2 = entry['text2']
            if return_pair:
                questions.append([q_1, q_2])
            else:
                questions.append(f"Sentence 1: {q_1} \nSentence 2: {q_2}")
            labels.append(int(entry['label']))
        return questions, labels
    train_texts, train_labels = process(dataset_train)
    test_texts, test_labels = process(dataset_test)
    return train_texts, train_labels, test_texts, test_labels

def load_sick(return_pair=False):
    dataset_train = load_dataset_huggingface("sick", split='train')
    dataset_test = load_dataset_huggingface("sick", split='test')
    def process(dataset):
        questions = []
        labels = []
        for entry in dataset:
            q_1 = entry['sentence_A']
            q_2 = entry['sentence_B']
            if return_pair:
                questions.append([q_1, q_2])
            else:
                questions.append(q_1.strip() + '\n' + 'question: ' + q_2)
            labels.append(int(entry['label']))
        return questions, labels
    train_texts, train_labels = process(dataset_train)
    test_texts, test_labels = process(dataset_test)
    return train_texts, train_labels, test_texts, test_labels

def load_wnli(return_pair=False):
    dataset_train = load_dataset_huggingface("glue", "wnli", split='train')
    dataset_test = load_dataset_huggingface("glue", "wnli", split='train')  # validation too small
    def process(dataset):
        questions = []
        labels = []
        for entry in dataset:
            q_1 = entry['sentence1']
            q_2 = entry['sentence2']
            if return_pair:
                questions.append([q_1, q_2])
            else:
                questions.append(q_1.strip() + '\n' + 'question: ' + q_2)
            labels.append(int(entry['label']))
        return questions, labels
    train_texts, train_labels = process(dataset_train)
    test_texts, test_labels = process(dataset_test)
    return train_texts, train_labels, test_texts, test_labels

def load_mr():
    train_texts, train_labels = read_csv_lmbff("./data/mr/train.csv")
    test_texts, test_labels = read_csv_lmbff("./data/mr/test.csv")
    return train_texts, train_labels, test_texts, test_labels

def load_cr():
    train_texts, train_labels = read_csv_lmbff("./data/cr/train.csv")
    test_texts, test_labels = read_csv_lmbff("./data/cr/test.csv")
    return train_texts, train_labels, test_texts, test_labels

def load_mpqa():
    train_texts, train_labels = read_csv_lmbff("./data/mpqa/train.csv")
    test_texts, test_labels = read_csv_lmbff("./data/mpqa/test.csv")
    return train_texts, train_labels, test_texts, test_labels

def load_cola():
    dataset_train = load_dataset_huggingface("glue", "cola", split='train')
    dataset_test = load_dataset_huggingface("glue", "cola", split='validation')
    train_texts, train_labels = get_texts_and_labels(dataset_train, key_text='sentence')
    test_texts, test_labels = get_texts_and_labels(dataset_test, key_text='sentence')
    return train_texts, train_labels, test_texts, test_labels

def load_tweet_hate():
    dataset_train = load_dataset_huggingface("tweet_eval", "hate", split='train')
    dataset_test = load_dataset_huggingface("tweet_eval", "hate", split='test')
    train_texts, train_labels = get_texts_and_labels(dataset_train)
    test_texts, test_labels = get_texts_and_labels(dataset_test)
    return train_texts, train_labels, test_texts, test_labels

def load_tweet(task='irony'):
    dataset_train = load_dataset_huggingface("tweet_eval", task, split='train')
    dataset_test = load_dataset_huggingface("tweet_eval", task, split='test')
    train_texts, train_labels = get_texts_and_labels(dataset_train)
    test_texts, test_labels = get_texts_and_labels(dataset_test)
    return train_texts, train_labels, test_texts, test_labels

def load_subj():
    dataset_train = load_dataset_huggingface("SetFit/subj", split='train')
    dataset_test = load_dataset_huggingface("SetFit/subj", split='test')
    train_texts, train_labels= get_texts_and_labels(dataset_train, 'text', 'label')
    test_texts, test_labels= get_texts_and_labels(dataset_test, 'text', 'label')
    return train_texts, train_labels, test_texts, test_labels

def load_sst2():
    def process_raw_data_sst(lines):
        """from lines in dataset to two lists of sentences and labels respectively"""
        labels = []
        sentences = []
        for line in lines:
            labels.append(int(line[0]))
            sentences.append(line[2:].strip())
        return sentences, labels

    with open(f"{ROOT_DIR}/data/sst2/stsa.binary.train", "r") as f:
        train_lines = f.readlines()
    with open(f"{ROOT_DIR}/data/sst2/stsa.binary.test", "r") as f:
        test_lines = f.readlines()
    train_sentences, train_labels = process_raw_data_sst(train_lines)
    test_sentences, test_labels = process_raw_data_sst(test_lines)
    return train_sentences, train_labels, test_sentences, test_labels

def load_agnews():
    train_data = pd.read_csv(f'{ROOT_DIR}/data/agnews/train.csv')
    test_data = pd.read_csv(f'{ROOT_DIR}/data/agnews/test.csv')

    train_sentences = train_data['Title'] + ". " + train_data['Description']
    train_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in train_sentences]) # some basic cleaning
    train_labels = list(train_data['Class Index'])
    test_sentences = test_data['Title'] + ". " + test_data['Description']
    test_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in test_sentences]) # some basic cleaning
    test_labels = list(test_data['Class Index']) 
    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4
    test_labels = [l - 1 for l in test_labels]
    return train_sentences, train_labels, test_sentences, test_labels

def load_trec():
    inv_label_dict = {'NUM': 0, 'LOC': 1, 'HUM': 2, 'DESC': 3, 'ENTY': 4, 'ABBR': 5}
    train_sentences = []
    train_labels = []
    with open(f'{ROOT_DIR}/data/trec/train.txt', 'r') as train_data:
        for line in train_data:
            train_label = line.split(' ')[0].split(':')[0]
            train_label = inv_label_dict[train_label]
            train_sentence = ' '.join(line.split(' ')[1:]).strip()
            # basic cleaning
            train_sentence = train_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            train_labels.append(train_label)
            train_sentences.append(train_sentence)

    test_sentences = []
    test_labels = []
    with open(f'{ROOT_DIR}/data/trec/test.txt', 'r') as test_data:
        for line in test_data:
            test_label = line.split(' ')[0].split(':')[0]
            test_label = inv_label_dict[test_label]
            test_sentence = ' '.join(line.split(' ')[1:]).strip()
            test_sentence = test_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            test_labels.append(test_label)
            test_sentences.append(test_sentence)
    return train_sentences, train_labels, test_sentences, test_labels

def get_cb(return_pair=False):
    train_questions = []
    train_answers = []
    with open(f"{ROOT_DIR}/data/cb/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            curr_label = myjson['label']
            if curr_label == 'contradiction':
                train_answers.append(0)
            elif curr_label == 'neutral':
                train_answers.append(1)
            elif curr_label == 'entailment':
                train_answers.append(2)
            # being a bit lazy here. We put the "question: " into the input and treat it like single sentence classification.
            if return_pair:
                train_questions.append([p.strip(), q])
            else:
                train_questions.append(p.strip() + '\n' + 'question: ' + q + '.')

    test_questions = []
    test_answers = []
    with open(f"{ROOT_DIR}/data/cb/val.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'contradiction':
                test_answers.append(0)
            elif myjson['label'] == 'neutral':
                test_answers.append(1)
            elif myjson['label'] == 'entailment':
                test_answers.append(2)
            else:
                exit('answer')
            if return_pair:
                test_questions.append([p.strip(), q])
            else:
                test_questions.append(p.strip() + '\n' + 'question: ' + q + '.')

    return train_questions, train_answers, test_questions, test_answers

def load_dbpedia():
    train_data = pd.read_csv(f'{ROOT_DIR}/data/dbpedia/train_subset.csv')
    test_data = pd.read_csv(f'{ROOT_DIR}/data/dbpedia/test.csv')

    train_sentences = train_data['Text']
    train_sentences = list([item.replace('""', '"') for item in train_sentences])
    train_labels = list(train_data['Class'])

    test_sentences = test_data['Text']
    test_sentences = list([item.replace('""', '"') for item in test_sentences])
    test_labels = list(test_data['Class'])
    
    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4...
    test_labels = [l - 1 for l in test_labels]
    return train_sentences, train_labels, test_sentences, test_labels

def load_rte(return_pair=False):
    train_questions = []
    train_answers = []
    with open("data/rte/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                train_answers.append(0)
            elif myjson['label'] == 'entailment':
                train_answers.append(1)
            else:
                exit('answer')
            if return_pair:
                train_questions.append([p, q])
            else:
                train_questions.append(p + '\n' + 'question: ' + q)

    test_questions = []
    test_answers = []
    with open("data/rte/val.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                test_answers.append(0)
            elif myjson['label'] == 'entailment':
                test_answers.append(1)
            else:
                exit('answer')
            if return_pair:
                test_questions.append([p, q])
            else:
                test_questions.append(p + '\n' + 'question: ' + q)
    return train_questions, train_answers, test_questions, test_answers

def load_dataset(params):
    """
    Load train and test data
    :param params: experiment parameter, which contains dataset spec
    :return: train_x, train_y, test_x, test_y
    """

    if params['dataset'] == 'sst2':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst2()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['negative'], 1: ['positive']}
        params['inv_label_dict'] = {'negative': 0, 'positive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'mr':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_mr()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['negative'], 1: ['positive']}
        params['inv_label_dict'] = {'negative': 0, 'positive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'cr':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_cr()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['negative'], 1: ['positive']}
        params['inv_label_dict'] = {'negative': 0, 'positive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'sst5':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst5()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['terrible'], 1: ['bad'], 2: ['okay'], 3: ['good'], 4: ['great']}
        params['inv_label_dict'] = {'terrible': 0, 'bad': 1, 'okay': 2, 'good': 3, 'great': 4}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'financial_phrasebank':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_financial_phrasebank()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Sentence: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['negative'], 1: ['neutral'], 2: ['positive']}
        params['inv_label_dict'] = {'negative': 0, 'neutral': 1, 'positive': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'poem_sentiment':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_poem_sentiment()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Verse text: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['negative'], 1: ['neutral'], 2: ['positive'], 3: ['mixed']}
        params['inv_label_dict'] = {'negative': 0, 'neutral': 1, 'positive': 2, 'mixed': 3}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'hate_speech18':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_hate_speech18()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Text: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['hate']}
        params['inv_label_dict'] = {'neutral': 0, 'hate': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'ethos_binary':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_ethos_binary()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Text: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['hate']}
        params['inv_label_dict'] = {'neutral': 0, 'hate': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'ethos_national_origin':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_ethos_national_origin()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Text: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['hate']}
        params['inv_label_dict'] = {'neutral': 0, 'hate': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'ethos_race':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_ethos_race()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Text: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['hate']}
        params['inv_label_dict'] = {'neutral': 0, 'hate': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'ethos_religion':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_ethos_religion()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Text: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['hate']}
        params['inv_label_dict'] = {'neutral': 0, 'hate': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'mpqa':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_mpqa()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['negative'], 1: ['positive']}
        params['inv_label_dict'] = {'negative': 0, 'positive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'sst2_instruction':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst2()
        params['prompt_prefix'] = "Classify the movie reviews into the categories of Positive and Negative.\n\n"
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
        params['inv_label_dict'] = {'Negative': 0, 'Positive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'cola':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_cola()
        params['prompt_prefix'] = "Are the following sentences grammatically correct?\n\n"
        params["q_prefix"] = "Sentence: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['Wrong'], 1: ['Correct']}
        params['inv_label_dict'] = {'Wrong': 0, 'Correct': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'tweet_hate':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_tweet_hate()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Tweet: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['hate']}
        params['inv_label_dict'] = {'neutral': 0, 'hate': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'tweet_hate_instruction':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_tweet_hate()
        params['prompt_prefix'] = "Classify tweets that are hateful against immigrants or women as hate and tweets that are not hateful against immigrants or women as neutral.\n\n"
        params["q_prefix"] = "Tweet: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['hate']}
        params['inv_label_dict'] = {'neutral': 0, 'hate': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'tweet_irony':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_tweet(task='irony')
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Tweet: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['ironic']}
        params['inv_label_dict'] = {'neutral': 0, 'ironic': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'tweet_irony_instruction':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_tweet(task='irony')
        params['prompt_prefix'] = "Classify tweets that are ironic as ironic, and tweets that are not ironic as neutral.\n\n"
        params["q_prefix"] = "Tweet: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['ironic']}
        params['inv_label_dict'] = {'neutral': 0, 'ironic': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'subj':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_subj()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Input: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['objective'], 1: ['subjective']}
        params['inv_label_dict'] = {'objective': 0, 'subjective': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'tweet_offensive':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_tweet(task='offensive')
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Tweet: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['offensive']}
        params['inv_label_dict'] = {'neutral': 0, 'offensive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'tweet_offensive_instruction':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_tweet(task='offensive')
        params['prompt_prefix'] = "Classify tweets that are offensive as offensive, and tweets that are not offensive as neutral.\n\n"
        params["q_prefix"] = "Tweet: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['offensive']}
        params['inv_label_dict'] = {'neutral': 0, 'offensive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'tweet_sentiment':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_tweet(task='sentiment')
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Tweet: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['Negative'], 1: ['Neutral'] , 2: ['Positive']}
        params['inv_label_dict'] = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'tweet_stance_climate':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_tweet(task='stance_climate')
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Tweet: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['none'], 1: ['against'] , 2: ['favor']}
        params['inv_label_dict'] = {'none': 0, 'against': 1, 'favor': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'tweet_stance_atheism':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_tweet(task='stance_atheism')
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Tweet: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['none'], 1: ['against'] , 2: ['favor']}
        params['inv_label_dict'] = {'none': 0, 'against': 1, 'favor': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'tweet_stance_feminist':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_tweet(task='stance_feminist')
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Tweet: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['none'], 1: ['against'] , 2: ['favor']}
        params['inv_label_dict'] = {'none': 0, 'against': 1, 'favor': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'agnews':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_agnews()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['world'], 1: ['sports'], 2: ['business'], 3: ['technology', 'science']}
        params['inv_label_dict'] = {'world': 0, 'sports': 1, 'business': 2, 'technology': 3, 'science': 3} # notice index start from 1 here
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'trec':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_trec()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer type: "
        params['label_dict'] = {0: ['number'], 1: ['location'], 2: ['person'], 3: ['description'], 4: ['entity'], 5: ['abbre']}
        params['inv_label_dict'] = {'number': 0, 'location': 1, 'person': 2, 'description': 3, 'entity': 4, 'abbre': 5}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'medical_questions_pairs_pair':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_medical_questions_pairs(return_pair=True)
        params['prompt_prefix'] = ""
        params['q1_prefix'] = "Question 1: "
        params['q2_prefix'] = "Question 2: "
        params["q_prefix"] = ""
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['different'], 1: ['similar']}
        params['inv_label_dict'] = {'different': 0, 'similar': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'mrpc':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_mrpc(return_pair=False)
        params['prompt_prefix'] = ""
        params["q_prefix"] = ""
        params["a_prefix"] = "question: different or equivalent? answer: "
        params['label_dict'] = {0: ['different'], 1: ['equivalent']}
        params['inv_label_dict'] = {'different': 0, 'equivalent': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'sick':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sick(return_pair=False)
        params['prompt_prefix'] = ""
        params["q_prefix"] = ""
        params["a_prefix"] = "entailment, neutral, or contradiction? answer: "
        params['label_dict_in_context'] = {0: ['entailment'], 1: ['neutral'], 2: ['contradiction']}
        params['label_dict'] = {0: ['entail'], 1: ['neutral'], 2: ['contradiction']}
        params['inv_label_dict'] = {'entail': 0, 'neutral': 1, 'contradiction': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'wnli':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_wnli(return_pair=False)
        params['prompt_prefix'] = ""
        params['q1_prefix'] = "Premise: "
        params['q2_prefix'] = "Hypothesis: "
        params["q_prefix"] = ""
        params["a_prefix"] = "True or False? answer: "
        params['label_dict'] = {0: ['False'], 1: ['True']}
        params['inv_label_dict'] = {'False': 0, 'True': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'rte':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_rte(return_pair=False)
        params['prompt_prefix'] = ""
        params["q_prefix"] = ""
        params["a_prefix"] = "True or False? answer: "
        params['label_dict'] = {0: ['False'], 1: ['True']}
        params['inv_label_dict'] = {'False': 0, 'True': 1}
        params['num_user_input'] = 2
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'rte_instruction':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_rte()
        params['prompt_prefix'] = "Determine if the answer entails or answers the question.\n\n"
        params["q_prefix"] = " "
        params["a_prefix"] = "answer: "
        params['label_dict'] = {0: ['False'], 1: ['True']}
        params['inv_label_dict'] = {'False': 0, 'True': 1}
        params['num_user_input'] = 2
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'cb':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = get_cb(return_pair=False)
        params['prompt_prefix'] = ""
        params["q_prefix"] = ""
        params["a_prefix"] = "true, false, or neither? answer: "
        params['label_dict'] = {0: ['false'], 1: ['neither'], 2: ['true']}
        params['inv_label_dict'] = {'false': 0, 'neither': 1, 'true': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'dbpedia':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_dbpedia()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Article type: "
        params['label_dict'] = {0: ['company'], 1: ['school'], 2: ['artist'], 3: ['athlete'], 4: ['politics'], 5: ['transportation'], 6: ['building'], 7: ['nature'], 8: ['village'], 9: ['animal'], 10: ['plant'], 11: ['album'], 12: ['film'], 13: ['book']}
        params['inv_label_dict'] = {'company': 0, 'school': 1, 'artist': 2, 'athlete': 3, 'politics': 4, 'transportation': 5, 'building': 6, 'nature': 7, 'village': 8, 'animal': 9, 'plant': 10, 'album': 11, 'film': 12, 'book': 13}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    else:
        raise NotImplementedError
    return orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels