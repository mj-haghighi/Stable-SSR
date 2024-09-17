import torch
from .funcs import weighted_knn
from .enums import SamplingStrategy

def select_samples(feature_bank, modified_label, args):
    if args.sampling_strategy == SamplingStrategy.BASE_LINE:
        return select_samples_base_line(feature_bank, modified_label, args)

def select_samples_base_line(feature_bank, modified_label, args):
    prediction_knn = weighted_knn(feature_bank, feature_bank, modified_label, args.num_classes, args.k, 10)  # temperature in weighted KNN
    vote_y = torch.gather(prediction_knn, 1, modified_label.view(-1, 1)).squeeze()
    vote_max = prediction_knn.max(dim=1)[0]
    right_score = vote_y / vote_max
    clean_candidate_ids = torch.where(right_score >= args.theta_s)[0]
    undecided_ids = torch.where(right_score < args.theta_s)[0]
    return clean_candidate_ids, undecided_ids
