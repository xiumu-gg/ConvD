import torch
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def eval_for_tail(eval_data, model, device, data, descending, raoit=0):
    hits = []
    hits_left = []
    hits_right = []
    ranks = []
    ranks_left = []
    ranks_right = []
    ent_rel_multi_t = data['entity_relation']['as_tail']
    ent_rel_multi_h = data['entity_relation']['as_head']
    for _ in range(10):  # need at most Hits@10
        hits.append([])
        hits_left.append([])
        hits_right.append([])

    for batch_idx, batch_data in enumerate(tqdm(eval_data)):
        eval_h = batch_data[0].to(device)
        eval_t = batch_data[1].to(device)
        eval_r = batch_data[2].to(device)
        _, pred = model(eval_h, eval_r)  # evaluate corruptions by replacing the object, i.e. tail entity
        _, pred1 = model(eval_t, eval_r, inverse=True)

        # need to filter out the entities ranking above the target entity that form a
        # true (head, tail) entity pair in train/valid/test data
        for i in range(eval_h.size(0)):
            # get all tail entities that form triples with eval_h[i] as the head entity and eval_r[i] as the relation
            filter_t = ent_rel_multi_t[eval_h[i].item()][eval_r[i].item()]
            filter_h = ent_rel_multi_h[eval_t[i].item()][eval_r[i].item()]

            pred_value = pred[i][eval_t[i].item()].item()
            pred_value1 = pred1[i][eval_h[i].item()].item()
            pred[i][filter_t] = 0.0
            pred1[i][filter_h] = 0.0
            pred[i][eval_t[i].item()] = pred_value
            pred1[i][eval_h[i].item()] = pred_value1

        _, index = torch.sort(pred, 1, descending=True)  # pred: (batch_size, ent_count)
        _, index1 = torch.sort(pred1, 1, descending=True)
        index = index.cpu().numpy()  # index: (batch_size)
        index1 = index1.cpu().numpy()

        for i in range(eval_h.size(0)):
            # find the rank of the target entities
            rank = np.where(index[i] == eval_t[i].item())[0][0]
            rank1 = np.where(index1[i] == eval_h[i].item())[0][0]

            # rank+1, since the rank starts with 1 not 0
            ranks_left.append(rank1 + 1)
            ranks_right.append(rank + 1)
            ranks.append(rank1 + 1)
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits_left[hits_level].append(1.0)
                else:
                    hits_left[hits_level].append(0.0)

    return hits, hits_left, ranks, ranks_left, ranks_right

def output_eval_tail(results, data_name):
    hits = np.array(results[0])
    # hits_left = np.array(results[1])
    # ranks = np.array(results[2])
    # ranks_left = np.array(results[3])
    ranks_right = np.array(results[4])
    # r_ranks = 1.0 / ranks  # compute reciprocal rank
    # r_ranks_left = 1.0 / ranks_left
    r_ranks_right = 1.0 / ranks_right



    # print Hits@10, Hits@3, Hits@1, MR (mean rank), and MRR (mean reciprocal rank)
    print('For %s data: Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f' % (data_name, hits[9].mean(), hits[2].mean(), hits[0].mean()))

    print('For %s data: MR=%.4f - MRR=%.4f' % (data_name, ranks_right.mean(), r_ranks_right.mean()))