import pathlib
import pandas as pd
import random
import io
import torch
from scipy.stats import kendalltau, spearmanr, pearsonr
from transformers import AutoTokenizer
from sMARE import calculate_sMARE
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
path_to_data = pathlib.Path('data')
# path_to_model = pathlib.Path('output/qpp-deberta-reverse-classification-2023-05-31_20-54-58') # this is my best model!

path_to_model = pathlib.Path('/mnt/data/sara-salamat/QPP-performanceprediction/output2/ablation_content+ppn-2024-05-07_15-30-12')

# qpp-deberta-(-)-classification-2023-05-16_17-21-49
# path_to_model = pathlib.Path('output/qpp-bert-reverse-classification-2023-05-29_21-39-34')

#### loading models
# with open(path_to_model / 'model_ep1.pkl', 'rb') as f:
#     pred_model = torch.load(f)
q_19 = pd.read_csv('/home/sara-salamat/Projects/QPP-performanceprediction/data/msmarco-test2019-queries.tsv', sep='\t', header=None) ## qid \t query
dl_19 = pd.read_csv('/home/sara-salamat/Projects/QPP-performanceprediction/data/dl2019_ndcg10', sep='\t', header=None) ## qid \t map

q_20 = pd.read_csv('/home/sara-salamat/Projects/QPP-performanceprediction/data/msmarco-test2020-queries.tsv', sep='\t', header=None) ## qid \t query
dl_20 = pd.read_csv('/home/sara-salamat/Projects/QPP-performanceprediction/data/dl2020_ndcg10', sep='\t', header=None) ## qid \t map

queries_numbers_19 = []
maps_19 = []


# fq_19 = pd.read_csv('/home/abbas/abbas_qpp/trec_data/2021/2021_queries', sep='\t', header=None)
# fdl_19 = pd.read_csv('/home/abbas/abbas_qpp/trec_data/2021/dl2021_ndcg10', sep='\t', header=None)
# fq_20 = pd.read_csv('/home/abbas/abbas_qpp/trec_data/2022/2022_queries', sep='\t', header=None)
# fdl_20 = pd.read_csv('/home/abbas/abbas_qpp/trec_data/2022/dl2022_ndcg10', sep='\t', header=None)

# q_19 = pd.concat([fq_19,fq_20])
# dl_19 = pd.concat([fdl_19,fdl_20])
# q_19 = fq_20
# dl_19 = fdl_20


for i in range(len(dl_19)):
    # if dl_19.iloc[i][1].isdigit():
    
    # print(type(dl_19.iloc[i][1]))
    if dl_19.iloc[i][1] == 'all':
        continue
    if type(dl_19.iloc[i][1])==str:
        queries_numbers_19.append(int(dl_19.iloc[i][1]))
        maps_19.append((int(dl_19.iloc[i][1]), dl_19.iloc[i][2]))
    else:
        queries_numbers_19.append(dl_19.iloc[i][1])
        maps_19.append((dl_19.iloc[i][1], dl_19.iloc[i][2]))
selected_queries_19 = q_19[q_19[0].isin(queries_numbers_19)]

queries_numbers_20 = []
maps_20 = []
for i in range(len(dl_20)):
    # if dl_19.iloc[i][1].isdigit():
    
    # print(type(dl_19.iloc[i][1]))
    if dl_20.iloc[i][1] == 'all':
        continue
    if type(dl_20.iloc[i][1])==str:
        queries_numbers_20.append(int(dl_20.iloc[i][1]))
        maps_20.append((int(dl_20.iloc[i][1]), dl_20.iloc[i][2]))
    else:
        queries_numbers_20.append(dl_20.iloc[i][1])
        maps_20.append((dl_20.iloc[i][1], dl_20.iloc[i][2]))
selected_queries_20 = q_20[q_20[0].isin(queries_numbers_20)] ## finding the queries

q_hard = pd.read_csv(path_to_data / 'hard_queries', sep='\t', header=None) ## qid \t query
dl_hard_eval = pd.read_csv(path_to_data / 'dlhard_ndcg10', sep='\t', header=None) ## qid \t map

queries_numbers_hard = []
maps_hard = []
for i in range(len(q_hard)):
    queries_numbers_hard.append(q_hard.iloc[i][0])
    maps_hard.append((dl_hard_eval.iloc[i][1], dl_hard_eval.iloc[i][2]))

dev_small = "/home/sara-salamat/Projects/QPP-performanceprediction/data/dev_small/queries.dev.small.tsv"
queries_dev_small = pd.read_csv(dev_small, sep='\t', header=None) ## qid \t query
org_scores_dev_small = pd.read_csv('/home/sara-salamat/Projects/QPP-performanceprediction/data/dev_small/dlmsmarco_mrr10', sep='\t', header=None) ## qid \s map
queries_numbers_dev_small = []
maps_dev_small = []
for i in range(len(org_scores_dev_small)):
    queries_numbers_dev_small.append(org_scores_dev_small.iloc[i][0])
    maps_dev_small.append((org_scores_dev_small.iloc[i][0], org_scores_dev_small.iloc[i][1]))
selected_queries_dev_small = queries_dev_small[queries_dev_small[0].isin(list(map(int, queries_numbers_dev_small)))] ## finding the queries


with open(path_to_model / 'model_ep12.pkl', 'rb') as f:
    pred_model = torch.load(f)

model_name = 'microsoft/deberta-v3-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
Task = 'classification'

def to_cuda(Tokenizer_output):
    tokens_tensor = Tokenizer_output['input_ids'].to('cuda')
    token_type_ids = Tokenizer_output['token_type_ids'].to('cuda')
    attention_mask = Tokenizer_output['attention_mask'].to('cuda')
    output = {'input_ids' : tokens_tensor, 
              'token_type_ids' : token_type_ids, 
              'attention_mask' : attention_mask}
    return output

similarity_function = torch.nn.CosineSimilarity()
def is_lower(a,b, dl):
    if dl=='19':
        selected_queries = selected_queries_19
    elif dl=='20':
        selected_queries = selected_queries_20
    elif dl=='hard':
        selected_queries = q_hard
    elif dl=='dev_small':
        selected_queries = selected_queries_dev_small
    q_a = selected_queries[selected_queries[0]==a][1].values[0]
    q_b = selected_queries[selected_queries[0]==b][1].values[0]

    
    predicted_score = pred_model.predict(
    to_cuda(tokenizer(q_a,return_tensors='pt')),
    to_cuda(tokenizer(q_b,return_tensors='pt')),
    )
    # a_difficulty = pred_model.get_difficulty_vec(to_cuda(tokenizer(q_a,return_tensors='pt')))
    # b_difficulty = pred_model.get_difficulty_vec(to_cuda(tokenizer(q_b,return_tensors='pt')))
    # print('Cosine similarity: ', similarity_function(a_difficulty,b_difficulty).item())
    predicted_score_v = pred_model.predict(
    to_cuda(tokenizer(q_b,return_tensors='pt')),
    to_cuda(tokenizer(q_a,return_tensors='pt')),
    )

    predicted_difference_score = 0.5*(predicted_score + 1 - predicted_score_v)# Formula
    # print('predicted_difference_score: ',predicted_difference_score)
    # predicted_difference_score = predicted_score
    if Task == 'regression':
        threshold = 0
    elif Task == 'classification':
        threshold = 0.5
    else: 
        print('Wrong task?')
    if  predicted_difference_score.item()<threshold:
        return True
    else:
        return False

def true_label(a,b,dl):
    if dl=='19':
        maps_dict = dict(maps_19)
    elif dl=='20':
        maps_dict = dict(maps_20)
    elif dl=='hard':
        maps_dict = dict(maps_hard)
    elif dl=='dev_small':
        maps_dict = dict(maps_dev_small)
    map_a = maps_dict[a]
    # print('map a: ', map_a)
    map_b = maps_dict[b]
    # print('map b: ', map_b)
    if map_a<map_b:
        return True
    else:
        return False

def _insort_right(a, x, q, counter, acc_counter,dl):
    """
    Insert item x in list a, and keep it sorted assuming a is sorted.
    If x is already in a, insert it to the right of the rightmost x.
    """
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo+hi)//2
        q += 1
        less = is_lower(x, a[mid],dl)
        
        accurate = true_label(x, a[mid],dl)
        counter = counter +1

        if accurate==less:
            acc_counter = acc_counter+1
        #     # print('accurate!')
        # else:
        #     # print('Not Accurate!')
        if less: hi = mid
        else: lo = mid+1
    a.insert(lo, x)
    return q , counter, acc_counter

def order(items, dl):
    ordered, q = [], 0
    counter = 0
    acc_counter = 0
    for item in items:
        q , counter, acc_counter = _insort_right(ordered, item, q, counter, acc_counter, dl)
    print('accuracy: ', acc_counter/counter)
    print('total comparisons: ', counter)
    return ordered, q

print('===== 1 =====')
print('DL19')
ordered_list, q = order(selected_queries_19[0].to_list(), '19')
sorted_q19 = list(sorted(maps_19, key=lambda x: x[1], reverse=False))
# actual_sorting = list(map(lambda x: int(x[0]),sorted_q19))
actual_sorting = list(map(lambda x: x[0],sorted_q19))
predicted_sorting = ordered_list
list_1 = []
list_2 = []
for qid in selected_queries_19[0]:
    list_1.append(actual_sorting.index(qid))
    list_2.append(predicted_sorting.index(qid))


print(kendalltau(list_1,list_2))
print(spearmanr(list_1, list_2))
print(pearsonr(list_1, list_2))
print('smare: ', calculate_sMARE(list_1, list_2))


print('DL20')
ordered_list, q = order(selected_queries_20[0].to_list(), dl='20')
sorted_q20 = list(sorted(maps_20, key=lambda x: x[1], reverse=False))
# actual_sorting = list(map(lambda x: int(x[0]),sorted_q19))
actual_sorting = list(map(lambda x: x[0],sorted_q20))
predicted_sorting = ordered_list
list_1 = []
list_2 = []
for qid in selected_queries_20[0]:
    list_1.append(actual_sorting.index(qid))
    list_2.append(predicted_sorting.index(qid))


print(kendalltau(list_1,list_2))
print(spearmanr(list_1, list_2))
print(pearsonr(list_1, list_2))
print('smare: ', calculate_sMARE(list_1, list_2))

print('DLhard')
ordered_list, q = order(q_hard[0].to_list(), dl='hard')
sorted_hard = list(sorted(maps_hard, key=lambda x: x[1], reverse=False))
# actual_sorting = list(map(lambda x: int(x[0]),sorted_q19))
actual_sorting = list(map(lambda x: x[0],sorted_hard))
predicted_sorting = ordered_list
list_1 = []
list_2 = []
for qid in q_hard[0]:
    list_1.append(actual_sorting.index(qid))
    list_2.append(predicted_sorting.index(qid))


print(kendalltau(list_1,list_2))
print(spearmanr(list_1, list_2))
print(pearsonr(list_1, list_2))
print('smare: ', calculate_sMARE(list_1, list_2))

print("Dev Small")

ordered_list_devsmall, q = order(selected_queries_dev_small[0].to_list(), dl='dev_small')
sorted_devsmall = list(sorted(maps_dev_small, key=lambda x: x[1], reverse=False))
actual_sorting_devsmall = list(map(lambda x: int(x[0]),sorted_devsmall))
predicted_sorting_devsmall = ordered_list_devsmall
list_1 = []
list_2 = []
for qid in selected_queries_dev_small[0]:
    list_1.append(actual_sorting_devsmall.index(qid))
    list_2.append(predicted_sorting_devsmall.index(qid))

print(kendalltau(list_1,list_2))
print(spearmanr(list_1, list_2))
print('smare: ', calculate_sMARE(list_1, list_2))
