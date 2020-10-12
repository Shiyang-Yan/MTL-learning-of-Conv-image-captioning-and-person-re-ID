# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from torch.nn import functional as F
from utils.reid_metric import R1_mAP
from torch.nn import functional as F
global ITER
ITER = 0

worddict_tmp = pickle.load(open('reid-data/wordlist_reid.p', 'rb'))
wordlist = [l for l in iter(worddict_tmp.keys()) if l != '</S>']
wordlist_final = ['EOS'] + sorted(wordlist)
max_tokens = 20



def get_grad_cos_sim(grad1, grad2):
    """Computes cos simillarity of gradients after flattening of tensors.
    """
    grad1 = torch.cat([x.data.view((-1,)).cpu() for x in grad1 if x is not None], 0).cpu()
    grad2 = torch.cat([x.data.view((-1,)).cpu() for x in grad2 if x is not None], 0).cpu()

   # grad1 = torch.tensor(grad1.data).view((-1,))
   # grad2 = torch.tensor(grad2.data).view((-1,))
    return F.cosine_similarity(grad1,grad2,dim = 0)
def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, language, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        language = language.to(device) if torch.cuda.device_count() >= 1 else language
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        #score, feat = model(img, language)
##########################################################3
        score, feat, wordact = model(img, language)
        print (wordact.size())
        wordact = wordact[:,:,:-1]
        wordclass_v = language[:,1:]
        wordact_t = wordact.permute(0, 2, 1).contiguous().view(\
           wordact.size(0)*(20-1), -1)
        wordclass_t = wordclass_v.contiguous().view(\
              wordact.size(0)*(20-1))

        loss_captioning = F.cross_entropy(wordact_t, \
               wordclass_t.contiguous())
#########################################################
        loss_reid = loss_fn(score, feat, target)
        loss = loss_reid + loss_captioning
        loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cetner_loss_weight,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img, language, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        language = language.to(device) if torch.cuda.device_count() >= 1 else language
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat, wordact = model(img, language)
        wordact = wordact[:,:,:-1]
        wordclass_v = language[:,1:]
        wordact_t = wordact.permute(0, 2, 1).contiguous().view(\
           wordact.size(0)*(20-1), -1)
        wordclass_t = wordclass_v.contiguous().view(\
              wordact.size(0)*(20-1))

        loss_captioning = F.cross_entropy(wordact_t, \
               wordclass_t.contiguous())

        loss_reid = loss_fn(score, feat, target)


            
        grad_reid = torch.autograd.grad(loss_reid, [value for para, value in model.named_parameters() if 'classifier' not in para and 'bottleneck' not in para if 'bn' not in para and 'att' not in para and 'capmodel' not in para], allow_unused=True, retain_graph = True)
        grad_captioning = torch.autograd.grad(loss_captioning, [value for para, value in model.named_parameters() if 'bn' not in para and 'classifier' not in para and 'att' not in para and 'capmodel' not in para and 'bottleneck' not in para], allow_unused=True, retain_graph = True)
        sim = get_grad_cos_sim(grad_reid, grad_captioning).cpu().numpy()


        lamb = (sim+1)/2
        loss = loss_reid + lamb * loss_captioning

        #print("Total loss is {}, caption loss is {}".format(loss, loss_captioning))
        loss.backward()
        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss_reid.item(), loss_captioning.item(), acc.item()

    return Engine(_update)


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
            batchsize = data.size(0)
            wordclass_feed = np.zeros((batchsize, max_tokens), dtype='int64')
            wordclass_feed[:,0] = wordlist_final.index('<S>')
            outcaps = np.empty((batchsize, 0)).tolist()

            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            language = language.to(device) if torch.cuda.device_count() >= 1 else language
            for j in range(max_tokens-1):
                wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda()
                features, wordact = model(data, wordclass)
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
         #   feat = model(data, language)
            return features, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)


def do_train_with_center(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer,
                                                                     'center_param': center_criterion,
                                                                     'optimizer_center': optimizer_center})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)
