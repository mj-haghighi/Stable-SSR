from tqdm import tqdm
import torch
import torch.nn.functional as F


def extract_features(dataloader, encoder, classifier=None):
    feature_bank = []
    preds_logits = []
    for (data, _, _) in tqdm(dataloader, desc='Feature extracting'):
        data = data.cuda()
        feature = encoder(data)
        feature_bank.append(feature)
        if classifier is not None:
            batch_pred_logits = classifier(feature)
            preds_logits.append(batch_pred_logits)
    feature_bank = F.normalize(torch.cat(feature_bank, dim=0), dim=1)
    return feature_bank, preds_logits



def extract_cifar_features(dataloader, encoder, classifier=None):
    feature_bank = []
    preds_logits = []
    for (data, _, _, _) in tqdm(dataloader, desc='Feature extracting'):
        data = data.cuda()
        feature = encoder(data)
        feature_bank.append(feature)
        if classifier is not None:
            batch_pred_logits = classifier(feature)
            preds_logits.append(batch_pred_logits)
    feature_bank = F.normalize(torch.cat(feature_bank, dim=0), dim=1)
    return feature_bank, preds_logits
