import numpy as np
import time
from copy import deepcopy
import os
import sys
import torch
import pickle
import openai
from tqdm import tqdm
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
import logging
from datetime import date
from scipy.stats import entropy
from english_words import english_words_set
from essential_generators import DocumentGenerator
from scipy.special import rel_entr

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', force_download=True)

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(ROOT_DIR, 'saved_results')
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
    print(f"mkdir at {SAVE_DIR} for saving results")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chunk_size_helper(params):
    # Set the batch size (the size of the chunks determines the batch size). Default to 4 for GPT-2 and 20 for OpenAI if
    # no batch size is specified.
    bs = params['bs']
    if bs is None:
        if 'gpt2' in params['model'] or 'gptj' in params['model']:
            return 1
        else:
            assert params['model'] in ['ada', 'babbage', 'curie', 'davinci', 'ada-beta', 'babbage-beta', 'curie-beta', 'text-davinci-002', 'text-davinci-001', 'text-davinci-003', 'text-curie-001']
            return 20
    else:
        return bs

def random_sampling(sentences, labels, num):
    """randomly sample subset of the training pairs"""
    assert len(sentences) == len(labels)
    if num > len(labels):
        assert False, f"you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}"
    idxs = np.random.choice(len(labels), size=num, replace=False)
    selected_sentences = [sentences[i] for i in idxs]
    selected_labels = [labels[i] for i in idxs]
    return deepcopy(selected_sentences), deepcopy(selected_labels)

gpt2_model = None
gpt2_tokenizer = None
def setup_gpt2(model_name):
    # load the GPT-2 model
    global gpt2_model
    global gpt2_tokenizer
    if gpt2_model is None:
        print("Setting up GPT-2 model")
        gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
        gpt2_model.eval().cuda()
        
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # to batch generation, we pad on the left and mask those positions out.
        gpt2_tokenizer.padding_side = "left"
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id
        print("Finished")

def setup_gptj():
    # load the GPT-J model, being a little bit lazy here, reused the naming of the setup_gpt2 function
    global gpt2_model
    global gpt2_tokenizer
    if gpt2_model is None:
        print("Setting up GPT-J model")
        gpt2_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", low_cpu_mem_usage=True, cache_dir='./cache')
        gpt2_model.eval().cuda()
        
        gpt2_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir='./cache')
        # to batch generation, we pad on the left and mask those positions out.
        gpt2_tokenizer.padding_side = "left"
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id
        print("Finished")

def setup_gpt3():
    # get OpenAI access key
    with open(os.path.join(ROOT_DIR, 'openai_key.txt'), 'r') as f:
        key = f.readline().strip()
        openai.api_key = key

def complete_gpt2(prompt, l=10, model_name='gpt2-xl', num_log_probs=None, echo=False):
    ''' This function runs GPT-2 locally but places the outputs into an json that looks just like the one
     provided by the OpenAI API. '''
    if isinstance(prompt, str):
        prompt = [prompt] # the code below assumes a list
    input_ids = gpt2_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    
    # greedily generate l tokens
    if l > 0:
        # the generate function can handle left padded inputs automatically in HF
        # total_sequences is now the input + possible generated output
        total_sequences = gpt2_model.generate(input_ids=input_ids['input_ids'].cuda(), attention_mask=input_ids['attention_mask'].cuda(), max_length=l + len(input_ids['input_ids'][0]), do_sample=False)
    else:
        assert echo == True and l == 0
        total_sequences = input_ids['input_ids'].cuda()

    # they want the probs of the top tokens
    if num_log_probs is not None:
        # we are left padding, so we need to adjust the position IDs
        attention_mask = (total_sequences != 50256).float()
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        # get the logits for the context and the next l tokens
        logits = gpt2_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids, return_dict=True).logits.detach().cpu()
        if not echo:
            # get the top tokens and probs for the generated l tokens
            probs = torch.softmax(logits[:,-l-1:], dim=2).cpu()
        else:
            # get the top tokens and probs for the context and the generated l tokens
            probs = torch.softmax(logits, dim=2).cpu()
        top_probs, top_tokens = torch.topk(probs, k=num_log_probs)
        logprobs = torch.log(probs)
        top_log_probs = torch.log(top_probs)

    # create the return value to resemble OpenAI
    return_json = {}
    choices = []
    for batch_id in range(len(prompt)):
        curr_json = {}
        # text is just the optional context and next l tokens
        if not echo:
            curr_json['text'] = gpt2_tokenizer.decode(total_sequences[batch_id][-l:], skip_special_tokens=True)
        else:
            curr_json['text'] = gpt2_tokenizer.decode(total_sequences[batch_id], skip_special_tokens=True)

        # fill the return json with the top tokens and probs to match the OpenAI return value.
        if num_log_probs is not None:
            curr_json['logprobs'] = {}
            curr_json['logprobs']['top_logprobs'] = []
            curr_json['logprobs']['token_logprobs'] = []
            curr_json['logprobs']['tokens'] = []
            if not echo:
                # cutoff the -1 here because the probs are shifted one over for LMs
                for current_element_top_log_probs, current_element_top_tokens in zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1]):
                    # tokens is a list of the top token at each position
                    curr_json['logprobs']['tokens'].append(gpt2_tokenizer.decode([current_element_top_tokens[0]]))
                    # token_logprobs is a list of the logprob of the top token at each position
                    curr_json['logprobs']['token_logprobs'].append(current_element_top_log_probs[0].item())
                    # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
                    temp = {}
                    for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                        temp[gpt2_tokenizer.decode(token.item())] = log_prob.item()
                    curr_json['logprobs']['top_logprobs'].append(temp)
            else:
                # same as not above but small tweaks
                # we add null to the front because for the GPT models, they have null probability for the first token
                # (for some reason they don't have an beginning of sentence token)
                curr_json['logprobs']['top_logprobs'].append('null')
                # cutoff the -1 here because the probs are shifted one over for LMs
                for index, (current_element_top_log_probs, current_element_top_tokens) in enumerate(zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1])):
                    # skip padding tokens
                    if total_sequences[batch_id][index].item() == 50256:
                        continue
                    temp = {}
                    for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                        temp[gpt2_tokenizer.decode(token.item())] = log_prob.item()
                    curr_json['logprobs']['top_logprobs'].append(temp)
                for index in range(len(probs[batch_id])):
                    curr_json['logprobs']['tokens'].append(gpt2_tokenizer.decode([total_sequences[batch_id][index]]))
                curr_json['logprobs']['token_logprobs'].append('null')
                for index, log_probs_token_position_j in enumerate(logprobs[batch_id][:-1]):
                    # probs are left shifted for LMs 
                    curr_json['logprobs']['token_logprobs'].append(log_probs_token_position_j[total_sequences[batch_id][index+1]])

        choices.append(curr_json)
    return_json['choices'] = choices
    return return_json

def complete_gpt3(prompt, l, model_name, temp=0, num_log_probs=None, echo=False, n=None):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            response = openai.Completion.create(engine=model_name, prompt=prompt, max_tokens=l, temperature=temp,
                                                logprobs=num_log_probs, echo=echo, stop='\n', n=n)
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False

            print("API error:", error)
            time.sleep(1)
    return response

def complete(prompt, l, model, temp=0, num_log_probs=None, echo=False, n=None):
    """complete the prompt using a language model"""
    assert l >= 0
    assert temp >= 0
    if 'gpt2' in model:
        assert n == None # unsupported at the moment
        assert temp == 0 # unsupported at the moment
        setup_gpt2(model)
        return complete_gpt2(prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo)
    elif 'gptj' in model:
        setup_gptj()
        return complete_gpt2(prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo)
    else:
        setup_gpt3()
        return complete_gpt3(prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo, n=n)

def construct_prompt(params, train_sentences, train_labels, test_sentence):
    """construct a single prompt to be fed into the model"""
    # special case when the user defines a custom prompt function. 
    if ('prompt_func' in params.keys()) and (params['prompt_func'] is not None):
        return params['prompt_func'](params, train_sentences, train_labels, test_sentence)

    # take the prompt template and fill in the training and test example
    prompt = params["prompt_prefix"]
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]
    for s, l in zip(train_sentences, train_labels):
        prompt += q_prefix
        if isinstance(s, str):
            prompt += s + "\n"
        else:
            prompt += params["q1_prefix"] + s[0] + "\n"
            prompt += params["q2_prefix"] + s[1] + "\n"
        if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): # integer labels for classification
            assert params['task_format'] == 'classification'
            if ('label_dict_in_context' in params.keys()) and (params['label_dict_in_context'] is not None):
                # if use different label names/description for the context and prediction
                l_str = params["label_dict_in_context"][l][0] if isinstance(params["label_dict_in_context"][l], list) else params["label_dict_in_context"][l]
            else:
                l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
        else:
            assert isinstance(l, str) # string labels
            assert params['task_format'] == 'qa'
            l_str = l

        prompt += a_prefix
        prompt += l_str + "\n\n"

    prompt += q_prefix
    if isinstance(test_sentence, str):
        prompt += test_sentence + "\n"
    else:
        prompt += params["q1_prefix"] + test_sentence[0] + "\n"
        prompt += params["q2_prefix"] + test_sentence[1] + "\n"
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
    return prompt

def get_model_response(params, train_sentences, train_labels, test_sentences, return_all_prompts=False,
                       num_tokens_to_predict_override=None, override_prompt=None):
    """
    Obtain model's responses on test sentences, given the training examples
    :param params: parameters for the experiment
    :param train_sentences: few-shot training sentences
    :param train_labels: few-shot training labels
    :param test_sentences: few-shot test sentences
    :param return_all_prompts: whether to return all the prompts
    :param num_tokens_to_predict_override: whether to override num token to predict
    :param override_prompt: whether to override prompt
    :return: a list of dictionaries
    """
    all_raw_answers = []

    # can optionally ignore the normal prompt and feed in a custom prompt (used for contextual calibration)
    if override_prompt is None:
        prompts = []
        for test_sentence in test_sentences:
            prompts.append(construct_prompt(params, train_sentences, train_labels, test_sentence))
    else:
        prompts = override_prompt

    chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
    for chunk_id, test_chunk_prompts in tqdm(enumerate(chunked_prompts)):
        if num_tokens_to_predict_override is not None:
            num_tokens_to_predict = num_tokens_to_predict_override
        else:
            num_tokens_to_predict = params['num_tokens_to_predict']
        resp = complete(test_chunk_prompts, num_tokens_to_predict, params['model'], num_log_probs=params['api_num_log_prob'])
        for answer_id, answer in enumerate(resp['choices']):
            all_raw_answers.append(answer)
    if return_all_prompts:
        return all_raw_answers, prompts
    else:
        return all_raw_answers

def load_pickle(params):
    # load saved results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    assert os.path.isfile(file_name), f"file does not exist: {file_name}"
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    print(f"Loaded data from {file_name}")
    return data

def save_pickle(params, data):
    # save results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    if os.path.isfile(file_name):
        print("WARNING! overwriting existing saved files")
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    print(f"Saved to {file_name}")
    return data

def print_results(tree, names=('Original Accuracy  ','Calibrated Accuracy'), logger=None):
    # print out all results
    root = deepcopy(tree)
    if logger is not None:
        print_func = logger.info
    else:
        print_func = print
    for dataset in root.keys():
        print_func(f"\n\nDataset: {dataset}")
        models_node = root[dataset]
        for model in models_node.keys():
            print_func(f"\nModel: {model}")
            num_shots_node = models_node[model]
            for num_shots in num_shots_node.keys():
                accuracies = np.array(list(num_shots_node[num_shots].values()))
                accuracies_mean = np.mean(accuracies, axis=0)
                accuracies_low = np.min(accuracies, axis=0)
                accuracies_high = np.max(accuracies, axis=0)
                accuracies_std = np.std(accuracies, axis=0)

                print_func(f"\n{num_shots}-shot, {len(accuracies)} seeds")
                for i, (m, l, h, s) in enumerate(zip(accuracies_mean, accuracies_low, accuracies_high, accuracies_std)):
                    print_func(f"{names[i]} | Mean: {m:.4f}, Low: {l:.4f}, High: {h:.4f}, Std: {s:.4f}")
                print_func("\n")

def print_results_with_f1(tree, logger=None, score_names=['accuracy', 'macro-f1', 'MI'],
                                             setting_names=["Original  ", 
                                                            "Calibrated", 
                                                            "Calibr_rt ",
                                                            "Chance    "]):
    # print out all results under all settings
    root = deepcopy(tree)
    if logger is not None:
        print_func = logger.info
    else:
        print_func = print
    for dataset in root.keys():
        print_func(f"\n\nDataset: {dataset}")
        models_node = root[dataset]
        for model in models_node.keys():
            print_func(f"\nModel: {model}")
            num_shots_node = models_node[model]
            for num_shots in num_shots_node.keys():
                scores = np.array(list(num_shots_node[num_shots].values()))
                n_score = scores.shape[1]
                names = score_names[:n_score]
                names = names[:n_score]
                print_func(f"\n{num_shots}-shot, {len(scores)} seeds")
                for score_id in range(n_score):
                    score = scores[:, score_id, :]
                    score_mean = np.mean(score, axis=0)
                    score_low = np.min(score, axis=0)
                    score_high = np.max(score, axis=0)
                    score_std = np.std(score, axis=0)
                    for i, (m, l, h, s) in enumerate(zip(score_mean, score_low, score_high, score_std)):
                        print_func(f"{setting_names[i]} {names[score_id]} | Mean: {m:.3f}, Low: {l:.3f}, High: {h:.3f}, Std: {s:.3f}")
                    print_func("\n")

def load_results(params_list):
    # load saved results from model
    result_tree = dict()
    for params in params_list:
        saved_result = load_pickle(params)
        keys = [params['dataset'], params['model'], params['num_shots']]
        node = result_tree # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]
        node[params['seed']] = saved_result['accuracies']
    print_results(result_tree)

def count_words(texts, labels, word, n_class=2):
    counter = np.zeros(n_class)
    for text, label in zip(texts, labels):
        words = text.lower().split()
        counter[label]+=words.count(word)
    return counter

def ceil(number):
    return int(number + 1)

def sample_random_texts(texts, n_sample=5, seed=0, random_type="random_in_domain_words", cf_tokens=['N/A', 'null', '[MASK]'], length_ratio=1):
    """
    Construct content-free texts for estimating the model's prior.
    @params:
    texts: task corpus
    @Supporting content_free text types
    Assume the average length of the input texts is L.
    1. random_type="content_free_token": use L pre-defined content-free tokens to construct content_free texts
    2. random_type="random_english_words": use L random English words to construct content_free texts
    3. random_type="random_sentence": use random English sentences as content_free texts
    4. random_type="random_in_domain_words": use L random words sampled from the task corpus to construct content_free texts
    """
    np.random.seed(seed)
    is_sentence_pair = isinstance(texts[0], list)
    gen = DocumentGenerator()
    if not is_sentence_pair:
        all_words = []
        text_lengths = []
        for text in texts:
            words = text.lower().split()
            all_words = all_words + words
            text_lengths.append(len(words))
        ave_length = int(np.mean(text_lengths))
        random_texts = []
        if random_type == 'content_free_token':
            for cf_token in cf_tokens:
                random_texts.append(" ".join([cf_token]*ceil(length_ratio *ave_length))+" .")
        else:
            for i in range(n_sample):
                if random_type == 'random_english_words':
                    random_texts.append(" ".join(np.random.choice(sorted(english_words_set), size=ceil(length_ratio * ave_length))) + " .")
                elif random_type == 'random_sentence':
                    random_texts.append(gen.sentence())
                else:
                    random_texts.append(" ".join(np.random.choice(all_words, size=ceil(length_ratio * ave_length)).tolist()) + " .")
    else:
        all_words_1 = []
        all_words_2 = []
        text_lengths_1 = []
        text_lengths_2 = []
        for text in texts:
            words_1 = text[0].lower().split()
            words_2 = text[1].lower().split()
            all_words_1 += words_1
            all_words_2 += words_2
            text_lengths_1.append(len(words_1))
            text_lengths_2.append(len(words_2))
        ave_l_1 = int(np.mean(text_lengths_1))
        ave_l_2 = int(np.mean(text_lengths_2))
        random_texts = []
        if random_type == 'content_free_token':
            for cf_token in cf_tokens:
                random_texts.append([" ".join([cf_token]*ceil(length_ratio * ave_l_1))+" .", " ".join([cf_token]*ceil(length_ratio * ave_l_2))+" ."])
        else:
            for i in range(n_sample):
                if random_type == 'random_english_words':
                    random_texts.append([" ".join(np.random.choice(sorted(english_words_set), size=ceil(length_ratio * ave_l_1))) + " .", " ".join(np.random.choice(sorted(english_words_set), size=ceil(length_ratio * ave_l_2))) + " ."])
                elif random_type == 'random_sentence':
                    random_texts.append([gen.sentence(), gen.sentence()])
                else:
                    random_texts.append([" ".join(np.random.choice(all_words_1, size=ceil(length_ratio * ave_l_1)).tolist()) + " .", " ".join(np.random.choice(all_words_2, size=ceil(length_ratio * ave_l_2)).tolist()) + " ."])
    return random_texts

def set_file_handler(logger, file_name):
    today = date.today().strftime("%b-%d-%Y")
    log_dir = os.path.join("./log", today)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_path = os.path.join(log_dir, file_name)
            
    fh = logging.FileHandler(filename=log_path, mode='a')
    formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def get_label_probs_from_resp(params, raw_resp):
    """Obtain model's label probability for each of the test examples. The returned prob is NOT normalized"""
    # num_classes = len(params['label_dict'])
    # approx = params['approx']
    # assert len(raw_resp) == len(test_sentences)

    # Fill in the labels that is in the top k prob
    all_label_probs = []
    all_missing_positions = []
    for i, ans in enumerate(raw_resp):
        top_logprobs = ans['logprobs']['top_logprobs'][0]  # [0] since we only ask for complete one more token
        # print(top_logprobs)
        label_probs = [0.0] * len(params['label_dict'].keys())
        for j, label_list in params['label_dict'].items():
            all_found = True
            for label in label_list:  # each possible label correspond to the same class
                label = " " + label  # notice prompt does not have space after 'A:'
                if label in top_logprobs:
                    label_probs[j] += np.exp(top_logprobs[label])
                else:
                    all_found = False
            if not all_found:
                position = (i, j) # (which test example, which label)
                all_missing_positions.append(position)
        all_label_probs.append(label_probs)
    all_label_probs = np.array(all_label_probs) # prob not normalized

    return all_label_probs # NOT NORMALIZED

def kl_div(p, q):
    return sum(rel_entr(p, q))

def truncate_sentence(texts, max_word=128):
    truncated_texts = []
    for text in texts:
        if len(text.split())>max_word:
            words = text.split()
            new_text = " ".join(words[:max_word])
        else:
            new_text = text
        truncated_texts.append(new_text)
    return truncated_texts

def calibrate_probs(all_label_probs, p_cf, do_calibration=True):
    num_classes = len(p_cf)

    W = np.linalg.inv(np.identity(num_classes) * p_cf)
    b = np.zeros([num_classes, 1])

    ans_label_list = []
    calibrated_prob_list = []
    for label_probs in all_label_probs:
        label_probs = label_probs / np.sum(label_probs) # normalize to 1
        if do_calibration:

            calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b
            calibrate_label_probs = calibrate_label_probs / np.sum(calibrate_label_probs)
            ans_label = np.argmax(calibrate_label_probs)
            calibrated_prob_list.append(calibrate_label_probs.flatten())
        else:
            ans_label = np.argmax(label_probs)
            calibrated_prob_list.append(label_probs)
        ans_label_list.append(ans_label)
    return np.array(ans_label_list), np.array(calibrated_prob_list)
    