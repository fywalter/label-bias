import argparse
from data_utils import load_dataset
from utils import *
from datetime import date
import logging
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

logger = logging.getLogger(__name__)

def set_file_handler(logger, file_name):
    today = date.today().strftime("%b-%d-%Y")
    log_base_dir = "./log"
    if not os.path.exists(log_base_dir):
        os.mkdir(log_base_dir)
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

def main(models, datasets, all_shots, num_seeds, subsample_test_set, api_num_log_prob, approx, use_saved_results, bs, seed, with_train=True, recompute_probs=False):
    """
    Run experiment or load past results, print accuracy

    with_train: whether to predict training set or not
    recompute_prob: whether to recompute the probs or use existing probs to accelerate inference
    """
    default_params = {
        'conditioned_on_correct_classes': True,
        'subsample_test_set': subsample_test_set,
        'api_num_log_prob': api_num_log_prob,
        'approx': approx,
        'bs': bs,
        'with_train': with_train,
        'recompute_probs': recompute_probs,
    }
    log_file_name = f"annotate_{'_'.join(models)}_{'_'.join(datasets)}.log"
    set_file_handler(logger, log_file_name)
    logger.info(f"all_shots={all_shots}, num_seeds={num_seeds}, with_train={with_train}")
    
    # list of all experiment parameters to run
    all_params = []
    for model in models:
        for dataset in datasets:
            for num_shots in all_shots:
                if seed is not None:    # if seed is passed, only run one seed
                    p = deepcopy(default_params)
                    p['model'] = model
                    p['dataset'] = dataset
                    p['seed'] = seed
                    p['num_shots'] = num_shots
                    p['expr_name'] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot_{repr(p['subsample_test_set'])}_subsample_seed{p['seed']}"
                    all_params.append(p)
                elif num_shots==0: # if zero-shot only one seed
                    p = deepcopy(default_params)
                    p['model'] = model
                    p['dataset'] = dataset
                    p['seed'] = 0
                    p['num_shots'] = num_shots
                    p['expr_name'] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot_{repr(p['subsample_test_set'])}_subsample_seed{p['seed']}"
                    all_params.append(p)
                else:
                    for s in range(num_seeds):
                        p = deepcopy(default_params)
                        p['model'] = model
                        p['dataset'] = dataset
                        p['seed'] = s
                        p['num_shots'] = num_shots
                        p['expr_name'] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot_{repr(p['subsample_test_set'])}_subsample_seed{p['seed']}"
                        all_params.append(p)

    logger.info(f"num of settings = {len(all_params)}")
    # query the model and save the responses
    if use_saved_results:
        load_results(all_params)
    else:
        save_results(all_params)


def save_results(params_list, freeze_test_set=True):
    """
    Run the model and save its responses and the rest of configs into a pickle file
    """
    result_tree = dict()
    for param_index, params in enumerate(params_list):
        logger.info(f"\nExperiment name:{params['expr_name']}")
        ### load data
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = load_dataset(params)
        is_sentence_pair = isinstance(all_train_sentences[0], list) # if sentence pair task or not, if sentence pair task, input is [text1, text2] rather than a string
        params_check(params) 

        ### truncate for ethos dataset one or two sentence are too long
        if 'ethos' in params['dataset']:
            logger.info("truncate sentences to max word num 128!!!")
            all_train_sentences = truncate_sentence(all_train_sentences)
            all_test_sentences = truncate_sentence(all_test_sentences)

        ### sample test set
        if params['subsample_test_set'] is None or len(all_test_labels) < params['subsample_test_set']:
            test_sentences, test_labels = all_test_sentences, all_test_labels
            logger.info(f"selecting full test set ({len(all_test_labels)} examples)")
        else:
            if freeze_test_set:
                np.random.seed(0) # always use seed 0 result if freeze
            else:
                np.random.seed(params['seed'])
            test_sentences, test_labels = random_sampling(all_test_sentences, all_test_labels, params['subsample_test_set'])
            logger.info(f"selecting {len(test_labels)} subsample of test set")

        ### sample few-shot training examples
        np.random.seed(params['seed'])
        train_sentences, train_labels = random_sampling(all_train_sentences, all_train_labels, params['num_shots'])
        empty_str = [" ", " "] if is_sentence_pair else " "
        context = construct_prompt(params, train_sentences, train_labels, empty_str)
        logger.info(context)
        
        if params['recompute_probs']:
            ### Evaluate the performance and save all results
            # obtaining model's response on test examples
            if params['with_train']:
                logger.info(f"getting raw resp for {len(all_train_sentences)} train sentences")
                raw_resp_train = get_model_response(params, train_sentences, train_labels, all_train_sentences)
                all_label_probs_train = get_label_probs(params, raw_resp_train, train_sentences, train_labels, all_train_sentences)
            else:
                raw_resp_train=[]
                all_label_probs_train=[]

            logger.info(f"getting raw resp for {len(test_sentences)} test sentences")
            raw_resp_test = get_model_response(params, train_sentences, train_labels, test_sentences)

            # get prob for each label
            all_label_probs = get_label_probs(params, raw_resp_test, train_sentences, train_labels, test_sentences)
        else:
            logger.info("Use existing probs!!!")
            saved_result = load_pickle(params)
            raw_resp_train = saved_result['raw_resp_train']
            all_label_probs_train = saved_result['all_label_probs_train']
            raw_resp_test = saved_result['raw_resp_test']
            all_label_probs = saved_result['all_label_probs']

        # calculate the estimated prior using predefined content-free tokens (contextual calibration)
        if is_sentence_pair:
            content_free_inputs = [["N/A", "N/A"], ["", ""], ["[MASK]", "[MASK]"]]
        else:
            content_free_inputs = ["N/A", "", "[MASK]"]
        p_cc, p_cc_resp = get_p_content_free(params, train_sentences, train_labels, content_free_inputs=content_free_inputs)

        # calibrate with random texts consists of random words sampled form the test corpus
        content_free_inputs = sample_random_texts(texts=test_sentences, n_sample=20, seed=params['seed'])
        logger.info(f"random texts for estimating prior: \n{content_free_inputs}")
        p_dc, p_dc_resp = get_p_content_free(params, train_sentences, train_labels, content_free_inputs=content_free_inputs)

        acc_original, f1_original = eval_accuracy(all_label_probs, test_labels)
        acc_calibrated, f1_calibrated = eval_accuracy(all_label_probs, test_labels, mode="diagonal_W", p_cf=p_cc)
        acc_calibrated_rt, f1_calibrated_rt = eval_accuracy(all_label_probs, test_labels, mode="diagonal_W", p_cf=p_dc)

        # chance performance
        n_test_samples = len(test_labels)
        label_set = set(test_labels)
        test_labels_chance = np.random.choice(list(label_set), size=n_test_samples)
        acc_chance = accuracy_score(test_labels, test_labels_chance)
        f1_chance = f1_score(test_labels, test_labels_chance, average='macro')

        accuracies = [acc_chance, acc_original, acc_calibrated, acc_calibrated_rt]
        f1s = [f1_chance, f1_original, f1_calibrated, f1_calibrated_rt]
        logger.info("             [score_chance, score_ori, score_cali, score_cali_dc]")
        logger.info(f"Accuracies: {accuracies}")
        logger.info(f"Macro f1s : {f1s}")
        logger.info(f"p_cc      : {p_cc}")
        logger.info(f"p_dc      : {p_dc}")
        # add to result_tree
        keys = [params['dataset'], params['model'], params['num_shots']]
        node = result_tree # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]
        node[params['seed']] = [accuracies, f1s]

        # save to file
        result_to_save = dict()
        params_to_save = deepcopy(params)
        result_to_save['params'] = params_to_save
        result_to_save['train_sentences'] = train_sentences
        result_to_save['train_labels'] = train_labels
        result_to_save['test_sentences'] = test_sentences
        result_to_save['test_labels'] = test_labels
        result_to_save['raw_resp_test'] = raw_resp_test
        result_to_save['all_label_probs'] = all_label_probs
        result_to_save['raw_resp_train'] = raw_resp_train
        result_to_save['all_label_probs_train'] = all_label_probs_train
        result_to_save['p_cc'] = p_cc
        result_to_save['p_cc_resp'] = p_cc_resp
        result_to_save['p_dc'] = p_dc
        result_to_save['p_dc_resp'] = p_dc_resp
        result_to_save['accuracies'] = accuracies
        result_to_save['f1s'] = f1s
        # result_to_save['mis'] = mis
        if 'prompt_func' in result_to_save['params'].keys():
            params_to_save['prompt_func'] = None
        save_pickle(params, result_to_save)

    setting_names = ["Chance    ", 
                     "Original  ",
                     "CC        ", 
                     "DC        "]
    print_results_with_f1(result_tree, logger=logger, score_names=['accuracy', 'macro-f1'],
                                                      setting_names=setting_names)

def eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False

    correctness_list = []
    assert len(all_label_probs) == len(test_labels)
    true_labels = []
    pred_labels = []
    for label_probs, true_label in zip(all_label_probs, test_labels):
        label_probs = label_probs / np.sum(label_probs) # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b

        ans_label = np.argmax(calibrate_label_probs)
        true_labels.append(true_label)
        pred_labels.append(ans_label)
        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    logger.info(f"Confusion matrix: \n {confusion_matrix(true_labels, pred_labels)}")
    return np.mean(correctness_list), f1_score(true_labels, pred_labels, average='macro')

def get_label_probs(params, raw_resp, train_sentences, train_labels, test_sentences):
    """Obtain model's label probability for each of the test examples. The returned prob is NOT normalized"""
    num_classes = len(params['label_dict'])
    approx = params['approx']
    assert len(raw_resp) == len(test_sentences)

    # Fill in the labels that is in the top k prob
    all_label_probs = []
    all_missing_positions = []
    for i, ans in enumerate(raw_resp):
        top_logprobs = ans['logprobs']['top_logprobs'][0]  # [0] since we only ask for complete one more token
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

    # Fill in the label probs that are NOT in top k probs, by asking the model to rate perplexity
    # This helps a lot in zero shot as most labels wil not be in Top 100 tokens returned by LM
    if (not approx) and (len(all_missing_positions) > 0):
        print(f"Missing probs: {len(all_missing_positions)}/{len(raw_resp) * num_classes}")
        all_additional_prompts = []
        num_prompts_each = []
        for position in all_missing_positions:
            which_sentence, which_label = position
            test_sentence = test_sentences[which_sentence]
            label_list = params['label_dict'][which_label]
            for label in label_list:
                prompt = construct_prompt(params, train_sentences, train_labels, test_sentence)
                prompt += " " + label
                all_additional_prompts.append(prompt)
            num_prompts_each.append(len(label_list))

        # chunk the prompts and feed into model
        chunked_prompts = list(chunks(all_additional_prompts, chunk_size_helper(params)))
        all_probs = []
        for chunk_id, chunk in enumerate(chunked_prompts):

            resp = complete(chunk, 0, params['model'], echo=True, num_log_probs=1)
            for ans in resp['choices']:
                prob = np.exp(ans['logprobs']['token_logprobs'][-1])

                all_probs.append(prob)

        assert sum(num_prompts_each) == len(all_probs)
        assert len(num_prompts_each) == len(all_missing_positions)

        # fill in corresponding entries in all_label_probs
        for index, num in enumerate(num_prompts_each):
            probs = []
            while num > 0:
                probs.append(all_probs.pop(0))
                num -= 1
            prob = np.sum(probs)

            i, j = all_missing_positions[index]
            all_label_probs[i][j] = prob

        assert len(all_probs) == 0, "all should be popped"
        assert (all_label_probs > 0).all(), "all should be populated with non-zero value"

    return all_label_probs # NOT NORMALIZED

def get_p_content_free(params, train_sentences, train_labels, content_free_inputs=('N/A',)):
    """Query model with content free input, return its prediction probability for each label"""
    label_dict = params['label_dict']

    all_p_y = []
    all_p_cf_resp = []

    num_log_probs = 2500 if 'gptj' in params['model'] else 100 
    logger.info(f"Number of logprob for content free: {num_log_probs}")
    for content_free_input in content_free_inputs:
        prompt = construct_prompt(params, train_sentences, train_labels, content_free_input)

        resp_all = complete(prompt, 1, params['model'], num_log_probs=num_log_probs)
        all_p_cf_resp.append(resp_all)
        p_y = [0] * len(label_dict)
        for i, answers in label_dict.items():
            prob = 0
            for a in answers:
                resp = complete(prompt + " " + a, 0, params['model'], echo=True, num_log_probs=1)
                prob += np.exp(resp['choices'][0]['logprobs']['token_logprobs'][-1])
            p_y[i] = prob
        all_p_y.append(p_y)

    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y) # normalize
    return p_y, all_p_cf_resp


def params_check(params):
    """sanity check the experiment params"""
    assert params['num_tokens_to_predict'] == 1
    # for classification, make sure that all of the class names are one word.
    for key, label_names in params['label_dict'].items():
        for label_id, label_name in enumerate(label_names):
            first_token_of_label_name = complete(' ' + label_name, 1, params['model'], echo=True, num_log_probs=2)['choices'][0]['logprobs']['tokens'][0]
            # print(first_token_of_label_name)
            if first_token_of_label_name[1:] != label_name:
                print('label name is more than 1 token', label_name)
                assert False

    if not (params['dataset'] in ['cb', 'rte']):
        # formatting: there should be a space after question/answer prefix
        assert len(params["q_prefix"]) == 0 or params["q_prefix"][-1] == " "
        assert len(params["a_prefix"]) == 0 or params["a_prefix"][-1] == " "
        assert len(params["prompt_prefix"]) == 0 or params["prompt_prefix"][-2:] == '\n\n'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--models', dest='models', action='store', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset(s), e.g., agnews')
    parser.add_argument('--num_seeds', dest='num_seeds', action='store', required=True, help='num seeds for the training set', type=int)
    parser.add_argument('--all_shots', dest='all_shots', action='store', required=True, help='num training examples to use')
    parser.add_argument(
        "--seed", type=int, default=None, help="if passed use this seed num"
    )
    # other arguments
    parser.add_argument('--subsample_test_set', dest='subsample_test_set', action='store', required=False, type=int,
                        default=None, help='size of test set to use to speed up eval. None means using all test set')
    parser.add_argument('--api_num_log_prob', dest='api_num_log_prob', action='store', required=False, type=int,
                        default=100, help='number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API')
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=None,
                        help='batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.')
    # flags
    parser.add_argument('--use_saved_results', dest='use_saved_results', action='store_const', const=True, default=False,
                        help='whether to load the results from pickle files and not run the model')
    parser.add_argument('--approx', dest='approx', action='store_const', const=True, default=False,
                        help='whether to set token prob to zero if not in top 100')
    parser.add_argument(
        "--with_train", action='store_const', const=True, default=False, help="whether to compute results on the training set"
    )
    parser.add_argument(
        "--recompute_probs", action='store_const', const=True, default=False, help="whether the recompute the raw prediction probabilities"
    )
    args = parser.parse_args()
    args = vars(args)

    # simple processing
    def convert_to_list(items, is_int=False):
        if is_int:
            return [int(s.strip()) for s in items.split(",")]
        else:
            return [s.strip() for s in items.split(",")]

    args['models'] = convert_to_list(args['models'])
    args['datasets'] = convert_to_list(args['datasets'])
    args['all_shots'] = convert_to_list(args['all_shots'], is_int=True)

    main(**args)