import torch
from utils.enums import RelabelingStrategy
from torch import tensor

def relabel_samples(preds_logits: tensor, human_labels, args):
    if args.relabeling_strategy == RelabelingStrategy.BASE_LINE:
        return relabel_samples_base_line(preds_logits, human_labels, args)
    
    raise Exception(f"Invalid relabeling strategy {args.relabeling_strategy}")

def relabel_samples_base_line(preds_logits: tensor, human_labels, args):
    preds_proba = torch.softmax(preds_logits, dim=1)
    his_score, his_label = preds_proba.max(1)

    human_labels_score = preds_proba[torch.arange(preds_logits.size()[0]), human_labels]
    relabeled_ids = torch.where((his_score > args.theta_r) & (his_label != human_labels))[0]
    
    modified_label = torch.clone(human_labels).detach()
    modified_label[relabeled_ids] = his_label[relabeled_ids]
    
    modified_score = torch.clone(his_score)
    modified_score[relabeled_ids] = his_score[relabeled_ids]
    
    return relabeled_ids, modified_label, modified_score, human_labels_score
