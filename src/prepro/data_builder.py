import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
from collections import Counter
from os.path import join as pjoin

import torch
from multiprocess import Pool

from others.logging import logger
from others.tokenization import BertTokenizer
from transformers import XLNetTokenizer

from others.utils import clean
from prepro.utils import _get_word_ngrams

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from tqdm import tqdm
import corenlp

# import stanza
# CORENLP_FOLDER = '/home/uoneway/stanford-corenlp-4.2.0'
# stanza.install_corenlp(dir=CORENLP_FOLDER)
# stanza.download_corenlp_models(model='english', version='4.2.0')
# from stanza.server import CoreNLPClient
# stanza.download('en', processors='tokenize')
# client = stanza.Pipeline('en', processors='tokenize')  # , use_gpu=False

# client = None

nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]


def recover_from_corenlp(s):
    s = re.sub(r' \'{\w}', '\'\g<1>', s)
    s = re.sub(r'\'\' {\w}', '\'\'\g<1>', s)



def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            tgt.append([])
            continue
        if (flag):
            tgt[-1].extend(tokens)
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt



def load_xml(p):
    tree = ET.parse(p)
    root = tree.getroot()
    title, byline, abs, paras = [], [], [], []
    title_node = list(root.iter('hedline'))
    if (len(title_node) > 0):
        try:
            title = [p.text.lower().split() for p in list(title_node[0].iter('hl1'))][0]
        except:
            print(p)

    else:
        return None, None
    byline_node = list(root.iter('byline'))
    byline_node = [n for n in byline_node if n.attrib['class'] == 'normalized_byline']
    if (len(byline_node) > 0):
        byline = byline_node[0].text.lower().split()
    abs_node = list(root.iter('abstract'))
    if (len(abs_node) > 0):
        try:
            abs = [p.text.lower().split() for p in list(abs_node[0].iter('p'))][0]
        except:
            print(p)

    else:
        return None, None
    abs = ' '.join(abs).split(';')
    abs[-1] = abs[-1].replace('(m)', '')
    abs[-1] = abs[-1].replace('(s)', '')

    for ww in nyt_remove_words:
        abs[-1] = abs[-1].replace('(' + ww + ')', '')
    abs = [p.split() for p in abs]
    abs = [p for p in abs if len(p) > 2]

    for doc_node in root.iter('block'):
        att = doc_node.get('class')
        # if(att == 'abstract'):
        #     abs = [p.text for p in list(f.iter('p'))]
        if (att == 'full_text'):
            paras = [p.text.lower().split() for p in list(doc_node.iter('p'))]
            break
    if (len(paras) > 0):
        if (len(byline) > 0):
            paras = [title + ['[unused3]'] + byline + ['[unused4]']] + paras
        else:
            paras = [title + ['[unused3]']] + paras

        return paras, abs
    else:
        return None, None


def tokenize(args):
    stories_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt


def format_to_bert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        print(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()


def _format_to_bert(params):
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']

        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                 is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data is None):
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(args):
    corpus_mapping = {}
    for corpus_type in ['valid', 'test', 'train']:
        temp = []
        for line in open(pjoin(args.map_path, 'mapping_' + corpus_type + '.txt')):
            temp.append(hashhex(line.strip()))
        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        real_name = f.split('/')[-1].split('.')[0]
        if (real_name in corpus_mapping['valid']):
            valid_files.append(f)
        elif (real_name in corpus_mapping['test']):
            test_files.append(f)
        elif (real_name in corpus_mapping['train']):
            train_files.append(f)
        # else:
        #     train_files.append(f)

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_to_lines(params):
    f, args = params
    print(f)
    source, tgt = load_json(f, args.lower)
    return {'src': source, 'tgt': tgt}




def format_xsum_to_lines(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'test', 'valid']

    corpus_mapping = json.load(open(pjoin(args.raw_path, 'XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json')))

    for corpus_type in datasets:
        mapped_fnames = corpus_mapping[corpus_type]
        root_src = pjoin(args.raw_path, 'restbody')
        root_tgt = pjoin(args.raw_path, 'firstsentence')
        # realnames = [fname.split('.')[0] for fname in os.listdir(root_src)]
        realnames = mapped_fnames

        a_lst = [(root_src, root_tgt, n) for n in realnames]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_xsum_to_lines, a_lst):
            if (d is None):
                continue
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_xsum_to_lines(params):
    src_path, root_tgt, name = params
    f_src = pjoin(src_path, name + '.restbody')
    f_tgt = pjoin(root_tgt, name + '.fs')
    if (os.path.exists(f_src) and os.path.exists(f_tgt)):
        print(name)
        source = []
        for sent in open(f_src):
            source.append(sent.split())
        tgt = []
        for sent in open(f_tgt):
            tgt.append(sent.split())
        return {'src': source, 'tgt': tgt}
    return None


def jsonls_to_dfs(from_dir, to_dir):
    jsonl_paths = get_file_paths(from_dir, suffix='.jsonl')
    os.makedirs(to_dir , exist_ok=True)

    for jsonl_path in tqdm(jsonl_paths):
        filename = os.path.splitext(os.path.basename(jsonl_path))[0]  ## extract file name
        df = _jsonl_to_df(jsonl_path)
        # df_selected = df[['title', 'text', 'summary']]
        # print(df.head())
        df.to_pickle(f"{to_dir}/{filename}_df.pickle")

def _jsonl_to_df(path):
    with open(path, 'r') as file:
        json_list = list(file)

        lines = []
        for json_str in json_list:
            line = json.loads(json_str)
            lines.append(line)

    df = pd.DataFrame(lines)
    return df

def get_file_paths(dir='./', suffix: str=''):
    file_paths = []
    filename_pattern = f'*{suffix}' if suffix != '' \
                    else '*' 

    file_paths = glob.glob(os.path.join(dir, filename_pattern))
    return file_paths


def dfs_to_jsons(from_dir, to_dir, n_cpus):
    df_paths = get_file_paths(from_dir, suffix='df.pickle')
    os.makedirs(to_dir, exist_ok=True)

    # n_cpus의 반 만큼 CoreNLP 서버 가동
    clients = [corenlp.CoreNLPClient(annotators=['tokenize','ssplit'], max_char_length=200000, 
            endpoint=f"http://localhost:900{port}") 
            for port in range(n_cpus // 2)]

    for df_file in df_paths:
        print(df_file)
        filename = os.path.splitext(os.path.basename(df_file))[0]  ## extract file name
        df = pd.read_pickle(df_file)
        _df_to_jsons(df, filename, to_dir, n_cpus, clients)

    for client in clients:
        client.stop()

def _df_to_jsons(df, prefix, to_dir, n_cpus, clients):

    NUM_DOCS_IN_ONE_FILE = 1000
    first_row_idx_list = list(range(0, len(df), NUM_DOCS_IN_ONE_FILE))
    digits_num = len(str(len(df)))
    client_list = clients * ((len(first_row_idx_list) // len(clients)) + 1)

    df_list = []
    file_name_list = []
    for i, first_row_idx in enumerate(first_row_idx_list):

        if i == len(first_row_idx_list) - 1:  # last element
            last_row_idx = len(df)
            df_list.append(df[first_row_idx:])
        else:
            last_row_idx = first_row_idx + NUM_DOCS_IN_ONE_FILE
            df_list.append(df[first_row_idx : last_row_idx])

        ## 정렬을 위해 파일이름 앞에 0 채워주기
        start_row_idx_str = (digits_num - len(str(first_row_idx)))*'0' + str(first_row_idx)
        last_row_idx_str = (digits_num - len(str(last_row_idx - 1)))*'0' + str(last_row_idx - 1)
        file_name = f'{to_dir}/{prefix}_{start_row_idx_str}_{last_row_idx_str}.json'
        file_name_list.append(file_name)
    
    print(f"----------{prefix} start({len(first_row_idx_list)} files)------------")
    # with Pool(processes=n_cpus) as pool:
    pool = Pool(n_cpus)
    port_idx = list(range(n_cpus))
    for result in pool.imap_unordered(_df_to_json, zip(df_list, file_name_list, client_list)):  # 길이가 작은거에 맞춰서 중단됨.
        print(result)
    pool.close()
    pool.join()
    # for params in tqdm(zip(dfs_split, idx_list, file_name_list)):
    #     _df_to_json(params)
    print(f"----------{prefix} end({len(first_row_idx_list)} files)------------")

def _df_to_json(params):
    df, file_name, client = params
    print(f"Start {file_name}")

    # with corenlp.CoreNLPClient(annotators=['tokenize','ssplit'], max_char_length=200000, endpoint=f"http://localhost:900{port}",) as client:  # , be_quiet=False , timeout=15000 
    # with CoreNLPClient(annotators=['tokenize','ssplit'], max_char_length=200000, threads=8, memory="8G", timeout=15000) as client:  # , be_quiet=False
    # global client

    json_list = []
    for i, row in df.iterrows():  # df.iloc[idx:end_idx].iterrows():
        text_len = len(row['text'])
        if text_len > 100000:
            print(f'Request is too long: {text_len} characters. Max length is 100000 characters.')
            continue
        try:
            tokenized_sents = client.annotate(row['text'])
            original_sents_list = []
            for sent in tokenized_sents.sentence:
                original_sents_list.append([token.word for token in sent.token])

            tokenized_sents = client.annotate(row['title'])
            query_sents_list = []
            for sent in tokenized_sents.sentence:
                query_sents_list.append([token.word for token in sent.token])

            tokenized_sents = client.annotate(row['summary'])
            summary_sents_list = []
            for sent in tokenized_sents.sentence:
                summary_sents_list.append([token.word for token in sent.token])

        except Exception as e:
            print(f'{file_name} 처리 중 에러 발생 \n{e}')
            continue

        json_list.append({'src': original_sents_list,
                        'query': query_sents_list,
                        'tgt': summary_sents_list
        })

    json_string = json.dumps(json_list, indent=4, ensure_ascii=False)
    #print(json_string)
    with open(file_name, 'w') as json_file:
        json_file.write(json_string)

    return f"End {file_name}"


def jsonl_to_bert(args):
    # Make dataset folders
    dataset_name = args.dataset
    dataset_root_dir = os.path.abspath( os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets'))
    print(dataset_root_dir)
    dataset_dir = os.path.join(dataset_root_dir, dataset_name)
    os.makedirs(dataset_dir , exist_ok=True)

    data_dir_names = ['raw', 'df', 'json', 'bert']
    data_dirs_dict = {data_dir_name: os.path.join(dataset_dir, data_dir_name) for data_dir_name in data_dir_names}
    for data_dir in data_dirs_dict.values():
        os.makedirs(data_dir, exist_ok=True)

    # Transform
    jsonls_to_dfs(data_dirs_dict['raw'], data_dirs_dict['df'])
    dfs_to_jsons(data_dirs_dict['df'], data_dirs_dict['json'], args.n_cpus)

    # json_to_bert(data_dirs_dict['json'], data_dirs_dict['bert'])


#     for sent in ann.sentence:
#         print([token.word for token in sent.token])
    

#         ## make json file
#         # 동일한 파일명 존재하면 덮어쓰는게 아니라 ignore됨에 따라 폴더 내 삭제 후 만들어주기
#         json_data_dir = f"{temp_dir}/{data_type}"
#         make_or_initial_dir(json_data_dir)
#         create_json_files(df, data_type=data_type, target_summary_sent=target_summary_sent, path=json_data_dir)
        
#         ## Convert json to bert.pt files
#         bert_data_dir = f"{to_dir}/{data_type}"
#         make_or_initial_dir(bert_data_dir)
        
#         os.system(f"python preprocess.py"
#             + f" -mode format_to_bert -dataset {data_type}"
#             + f" -raw_path {json_data_dir}"
#             + f" -save_path {bert_data_dir}"
#             + f" -log_file {log_file}"
#             + f" -lower -n_cpus {n_cpus}")
