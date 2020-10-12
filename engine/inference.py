# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
import torch.nn as nn
from ignite.engine import Engine
import pickle
from utils.reid_metric import R1_mAP, R1_mAP_reranking
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
worddict_tmp = pickle.load(open('/home/ECIT.QUB.AC.UK/3054256/Code/reid_strong/reid-strong-baseline/reid_data/wordlist_reid.p', 'rb'))
wordlist = [l for l in iter(worddict_tmp.keys()) if l != '</S>']
wordlist_final = ['EOS'] + sorted(wordlist)
max_tokens = 20
def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, language, pids, camids = batch
            batchsize = language.size(0)
            wordclass_feed = np.zeros((batchsize, max_tokens), dtype='int64')
            wordclass_feed[:,0] = wordlist_final.index('<S>')
            outcaps = np.empty((batchsize, 0)).tolist()

            data = data.to(device) if torch.cuda.device_count() >= 1 else data
           # language = language.to(device) if torch.cuda.device_count() >= 1 else language

            for j in range(max_tokens-1):
                wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda()
                features, wordact, _= model(data, wordclass)
                wordact = wordact[:,:,:-1]
                wordact_t = wordact.permute(0, 2, 1).contiguous().view(batchsize*(max_tokens-1), -1)
                wordprobs = F.softmax(wordact_t).cpu().data.numpy()
                wordids = np.argmax(wordprobs, axis=1)
                for k in range(batchsize):
                    word = wordlist_final[wordids[j+k*(max_tokens-1)]]
                    outcaps[k].append(word)
                    if(j < max_tokens-1):
                        wordclass_feed[k, j+1] = wordids[j+k*(max_tokens-1)]
            for j in range(batchsize):
                num_words = len(outcaps[j]) 
                if 'EOS' in outcaps[j]:
                    num_words = outcaps[j].index('EOS')
                outcap = ' '.join(outcaps[j][:num_words])
            feat, _, _ = model(data, wordclass)
            print (outcap)
        return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
       logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
