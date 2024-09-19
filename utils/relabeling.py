import torch
from utils.enums import RelabelingStrategy
from torch import tensor

def relabel_samples(preds_logits: tensor, human_labels, args, stability_score, human_labels_confidnce_score):
    if args.relabeling_strategy == RelabelingStrategy.BASE_LINE:
        return relabel_samples_base_line(preds_logits, human_labels, args)
    elif args.relabeling_strategy == RelabelingStrategy.STABLE:
        return relabel_samples_stable_score(
            preds_logits, human_labels, args,
            stability_score, human_labels_confidnce_score   
        )
    
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

# prediction_cls[torch.arange(pred_score.size()[0]), human_labels]
def relabel_samples_stable_score(preds_logits: tensor, human_labels, args, stability_score, human_labels_confidnce_score):
    preds_proba = torch.softmax(preds_logits, dim=1)
    his_score, his_label = preds_proba.max(1)

    human_labels_score = preds_proba[torch.arange(preds_logits.size()[0]), human_labels]

    relabeled_ids = torch.tensor([], dtype=int)
    print('stability_score.size(): ', stability_score.size())
    if stability_score.size()[0] > 0:
        relabeled_ids = torch.where((stability_score >= (args.theta_r * torch.max(stability_score).detach().item())) & (human_labels_confidnce_score < (1.0 / args.num_classes)))[0]
    
    modified_label = torch.clone(human_labels).detach()
    modified_label[relabeled_ids] = his_label[relabeled_ids]
    
    modified_score = torch.clone(human_labels_score)
    modified_score[relabeled_ids] = his_score[relabeled_ids]
    
    return relabeled_ids, modified_label, modified_score, human_labels_score
