import sys
import os
import torch
import random
import numpy as np
import time
import six
from config import get_parameter
from sklearn import metrics
import torch.nn.functional as F
from load_data import prepare_dataset
from transformers import BertTokenizer,AdamW
from ABSA_dataset_v3 import ABSA_Dataset,ABSA_collate_fn
from MCDG7 import MCDG
from torch import nn, optim
from SLR import ModifiedSupConLoss
from SLR import SupConLoss
from itertools import product
def set_random_seed(args):
    # set random seed
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def evaluate(model, dataloader, args):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    predictions, labels = [], []

    val_loss, val_acc = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
        # model.eval()
        # with torch.no_grad():
            length_, bert_length_, word_mapback, adj_oneshot, \
                adj_oneshot1, adj_oneshot2, \
                bert_segments_ids_list, word_mapback_N_S_list, \
                word_mapback_sub_list, bert_sub_sequence_list, \
                bert_N_S_sequence_list, aspect_masks, \
                bert_tokens, adj_oneshot3, max_node_num, bert_N_S_length_list, bert_sub_length_list, bert_N_S_segments_ids_list, bert_sub_segments_ids_list,adj_oneshot4,aspect_in_sub_list,adj_oneshot5,all_aspect_adj,all_aspect_label_adj,all_aspect_label_adj_reverse,all_aspect_label_adj_aa,all_clause_adj,all_clause_label_adj,all_clause_re_label_adj,all_clause_super_label_adj,all_clause_super_re_label_adj,labels_ = batch
            batch = (length_.to('cuda'), bert_length_.to('cuda'), word_mapback.to('cuda'), adj_oneshot.to('cuda'),adj_oneshot1.to('cuda'), adj_oneshot2.to('cuda'),bert_segments_ids_list.to('cuda'), word_mapback_N_S_list.to('cuda'),word_mapback_sub_list.to('cuda'), bert_sub_sequence_list.to('cuda'),bert_N_S_sequence_list.to('cuda'), aspect_masks.to('cuda'),bert_tokens.to('cuda'), adj_oneshot3.to('cuda'), max_node_num.to('cuda'), bert_N_S_length_list.to('cuda'), bert_sub_length_list.to('cuda'), bert_N_S_segments_ids_list.to('cuda'), bert_sub_segments_ids_list.to('cuda'),adj_oneshot4.to('cuda'),aspect_in_sub_list.to('cuda'),adj_oneshot5.to('cuda'),all_aspect_adj.to('cuda'),all_aspect_label_adj.to('cuda'),all_aspect_label_adj_reverse.to('cuda'),all_aspect_label_adj_aa.to('cuda'),all_clause_adj.to('cuda'),all_clause_label_adj.to('cuda'),all_clause_re_label_adj.to('cuda'),all_clause_super_label_adj.to('cuda'),all_clause_super_re_label_adj.to('cuda'),labels_.to('cuda'))
            inputs = batch
            label = labels_.to('cuda')

            logits,_,_ = model(inputs)
            logits = logits.squeeze(1)
            loss = F.cross_entropy(logits, label, reduction='mean')
            val_loss += loss.data
            labels1 = label
            bert_tokens = bert_tokens.to('cuda')
            predictions += np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
            labels += label.data.cpu().numpy().tolist()
            # pred = logits.argmax(dim=-1)
            # for index,i in enumerate(bert_tokens[pred!=labels1]):
            #     tokens = tokenizer.convert_ids_to_tokens(i)
            #     # print(tokens)
            #     text = ''
            #     sum = 0
            #     for j in tokens:
            #         if j == '[CLS]':
            #             continue
            #         elif j == '[SEP]':
            #             if sum == 0:
            #                 textprint = text
            #             else:
            #                 aspectprint = text
            #             text = ''
            #             sum = 1
            #         elif j == '[PAD]':
            #             break
            #         else:
            #             text = text + j + ' '
            #     print(textprint,aspectprint,pred[pred!=labels1][index],labels1[pred!=labels1][index])
    val_acc = metrics.accuracy_score(labels, predictions) * 100.0
    f1_score = metrics.f1_score(labels, predictions, average='macro')

    return val_loss / len(dataloader), val_acc, f1_score
def has_opposite_labels(stacked_tensor):
    count_0 = torch.sum(stacked_tensor == 0).item()
    count_1 = torch.sum(stacked_tensor == 1).item()
    count_2 = torch.sum(stacked_tensor == 2).item()
    if count_0 == 0:
        if count_2 <= 1 or count_1 <= 1:
            result = True
        else:
            result = False
    elif count_1 == 0:
        if count_2 <= 1 or count_0 <= 1:
            result = True
        else:
            result = False
    else:
        if count_2 <= 1 or count_1 <= 1 or count_0 <= 1:
            result = True
        else:
            result = False
    return not result

def train(args, train_dataloader, valid_dataloader, test_dataloader, model, optimizer,x):
    ############################################################
    # train
    print("Training Set: {}".format(len(train_dataloader)))
    print("Valid Set: {}".format(len(valid_dataloader)))
    print("Test Set: {}".format(len(test_dataloader)))
    similar_criterion = SupConLoss()
    # similar_criterion = ModifiedSupConLoss()
    train_acc_history, train_loss_history = [0.0], [0.0]
    val_acc_history, val_history, val_f1_score_history = [0.0], [0.0], [0.0]

    in_test_epoch, in_test_acc, in_test_f1 = 0, 0.0, 0.0
    patience = 0
    for epoch in range(args.epoch):
        begin_time = time.time()

        print("Epoch {}".format(epoch) + "-" * 60)

        train_loss, train_acc, train_step = 0.0, 0.0, 0

        train_all_predict = 0
        train_all_correct = 0

        for i, batch in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            length_, bert_length_, word_mapback, adj_oneshot, \
                adj_oneshot1, adj_oneshot2, \
                bert_segments_ids_list, word_mapback_N_S_list, \
                word_mapback_sub_list, bert_sub_sequence_list, \
                bert_N_S_sequence_list, aspect_masks, \
                bert_tokens, adj_oneshot3, max_node_num, bert_N_S_length_list, bert_sub_length_list, bert_N_S_segments_ids_list, bert_sub_segments_ids_list,adj_oneshot4,aspect_in_sub_list,adj_oneshot5,all_aspect_adj,all_aspect_label_adj,all_aspect_label_adj_reverse,all_aspect_label_adj_aa,all_clause_adj,all_clause_label_adj,all_clause_re_label_adj,all_clause_super_label_adj,all_clause_super_re_label_adj,labels_ = batch
            batch = (length_.to('cuda'), bert_length_.to('cuda'), word_mapback.to('cuda'), adj_oneshot.to('cuda'),adj_oneshot1.to('cuda'), adj_oneshot2.to('cuda'),bert_segments_ids_list.to('cuda'), word_mapback_N_S_list.to('cuda'),word_mapback_sub_list.to('cuda'), bert_sub_sequence_list.to('cuda'),bert_N_S_sequence_list.to('cuda'), aspect_masks.to('cuda'),bert_tokens.to('cuda'), adj_oneshot3.to('cuda'), max_node_num.to('cuda'), bert_N_S_length_list.to('cuda'), bert_sub_length_list.to('cuda'), bert_N_S_segments_ids_list.to('cuda'), bert_sub_segments_ids_list.to('cuda'),adj_oneshot4.to('cuda'),aspect_in_sub_list.to('cuda'),adj_oneshot5.to('cuda'),all_aspect_adj.to('cuda'),all_aspect_label_adj.to('cuda'),all_aspect_label_adj_reverse.to('cuda'),all_aspect_label_adj_aa.to('cuda'),all_clause_adj.to('cuda'),all_clause_label_adj.to('cuda'),all_clause_re_label_adj.to('cuda'),all_clause_super_label_adj.to('cuda'),all_clause_super_re_label_adj.to('cuda'),labels_.to('cuda'))
            inputs = batch
            label = labels_.to('cuda')

            logits,sim_loss,hidden = model(inputs)
            logits = logits.squeeze(1)
            loss = F.cross_entropy(logits, label, reduction='mean')
            # label1 = label
            # if has_opposite_labels(label1):
            #     normed_cls_hidden = F.normalize(hidden, dim=-1)
            #     similar_loss = similar_criterion(normed_cls_hidden.unsqueeze(1), labels=label1)
            #     loss = loss + 0.5 * similar_loss + 0.5* sim_loss
            # else:
            #     loss = loss+ 0.8 * sim_loss
            loss = loss + x * sim_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()

            train_all_predict += label.size()[0]
            train_all_correct += corrects.item()

            train_step += 1
            if train_step % args.log_step == 0:
                print('{}/{} train_loss:{:.6f}, train_acc:{:.4f}'.format(
                    i, len(train_dataloader), train_loss / train_step, 100.0 * train_all_correct / train_all_predict
                ))

        train_acc = 100.0 * train_all_correct / train_all_predict
        val_loss, val_acc, val_f1 = evaluate(model, test_dataloader, args)
        print(
            "[{:.2f}s] Pass!\nEnd of {} train_loss: {:.8f}, train_acc: {:.8f}, val_loss: {:.8f}, val_acc: {:.8f}, f1_score: {:.8f}".format(
                time.time() - begin_time, epoch, train_loss / train_step, train_acc, val_loss, val_acc, val_f1
            )
        )
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss / train_step)
        if val_acc > max(val_acc_history):
            model_file = "X_{}_A_{}_B_{}_epoch_{}_acc_{:.8f}_f1_{:.8f}_loss_{:.8f}.pt".format(
                x,args.A,args.B,epoch, val_acc, val_f1, val_loss)
            torch.save(model.state_dict(), os.path.join('result/laptop1', model_file))
        val_acc_history.append(float(val_acc))
        val_f1_score_history.append(val_f1)

    print('Training ended with {} epoches.'.format(epoch))
    _, last_test_acc, last_test_f1 = evaluate(model, test_dataloader, args)

    print('In Results: test_epoch:{}, test_acc:{}, test_f1:{}'.format(in_test_epoch, in_test_acc, in_test_f1))
    print('Last In Results: test_epoch:{}, test_acc:{}, test_f1:{}'.format(epoch, last_test_acc, last_test_f1))
def build_optimizer(config, model):
    lr = config.learning_rate
    weight_decay = config.weight_decay
    opt = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': AdamW,
        'adagrad': optim.Adagrad,
    }
    if 'momentum' in config:
        optimizer = opt[config.optimizer](
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=config['momentum']
        )
    else:
        optimizer = opt[config.optimizer](
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    return optimizer
def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')

def run(args, tokenizer):
    print_arguments(args)
    # train_path = 'laptop_train_v1.pkl'
    # test_path = 'laptop_test_v1.pkl'
    # vaild_path = 'laptop_test_v1.pkl'
    # train_path = 'res_train_v1.pkl'
    # test_path = 'res_test_v1.pkl'
    # vaild_path = 'res_test_v1.pkl'
    # train_path = 'laptop_train.pkl'
    # test_path = 'laptop_test.pkl'
    # vaild_path = 'laptop_test.pkl'
    train_path = 'res15_train.pkl'
    test_path = 'res15_test.pkl'
    vaild_path = 'res15_test.pkl'
    train_path1 = 'res15_train.pkl'
    test_path1 = 'res15_test.pkl'
    vaild_path1 = 'res15_test.pkl'

    # train_path = 'res_train.pkl'
    # test_path = 'res_test.pkl'
    # vaild_path = 'res_test.pkl'
    # train_path = 'tw_train.pkl'
    # test_path = 'tw_test.pkl'
    # vaild_path = 'tw_test.pkl'
    ###########################################################
    # data
    # train_dataloader, valid_dataloader, test_dataloader = prepare_dataset(train_path,test_path,vaild_path,args,ABSA_Dataset, ABSA_collate_fn)
    # args.A = 0.3
    # args.B = 0.7
    ###########################################################
    # model
    # model = MCDG(args).to(device=args.device)
    # model.load_state_dict(torch.load('result/laptop/epoch_1_acc_83.22884013_f1_0.80569697_loss_0.48146486.pt'))
    # print(model)

    ###########################################################
    # optimizer
    # optimizer = build_optimizer(args, model)
    # train(args, train_dataloader, valid_dataloader, test_dataloader, model, optimizer)
    # _, last_test_acc, last_test_f1= evaluate(model, test_dataloader, args)
    # print(last_test_acc,last_test_f1)
    x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    x1 = [0.5,0.3,0.7,0.4,0.6]
    x2 = [0.5,0.7,0.3,0.4,0.6,0.2]
    all_combinations = list(product(x1, x2))
    # train_dataloader, valid_dataloader, test_dataloader = prepare_dataset(train_path,test_path,vaild_path,args,ABSA_Dataset, ABSA_collate_fn)
    for j1 in range(1):
        if j1 != 0:
            train_path = 'res15_train.pkl'
            test_path = 'res15_test.pkl'
            vaild_path = 'res15_test.pkl'
        else:
            train_path = 'res16_train.pkl'
            test_path = 'res16_test.pkl'
            vaild_path = 'res16_test.pkl'
        train_dataloader, valid_dataloader, test_dataloader = prepare_dataset(train_path, test_path, vaild_path, args,
                                                                              ABSA_Dataset, ABSA_collate_fn)
        for j in x:
            for i in all_combinations:
                if i[0] ==0.5 and i[1] == 0.3 and j == 0.7:
                # if i[0] + i[1] == 1 or i[0] + i[1] != 1:
                    torch.cuda.empty_cache()
                    args.A = i[0]
                    args.B = i[1]
                    model = MCDG(args).to(device=args.device)
                    model.load_state_dict(
                        torch.load('result/laptop1/X_0.7_A_0.5_B_0.3_epoch_5_acc_93.01948052_f1_0.80101663_loss_0.63473666.pt'))
                    # print(model)
                    # _, last_test_acc, last_test_f1= evaluate(model, test_dataloader, args)
                    # print(last_test_acc,last_test_f1)
                    optimizer = build_optimizer(args, model)
                    train(args, train_dataloader, valid_dataloader, test_dataloader, model, optimizer, j)


if __name__ == '__main__':
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    args = get_parameter()
    set_random_seed(args)

    run(args, tokenizer=bert_tokenizer)