import torch
from .funcs import weighted_knn
from .enums import SamplingStrategy

def select_samples(feature_bank, modified_label, args, stability_score):
    if args.sampling_strategy == SamplingStrategy.BASE_LINE:
        return select_samples_base_line(feature_bank, modified_label, args)
    elif args.sampling_strategy == SamplingStrategy.STABLE_EXTENDED_EXCLUDED:
        return select_samples_extended_excluded(feature_bank, modified_label, args, stability_score)
    raise Exception("selection strategy is not valid")

def select_samples_base_line(feature_bank, modified_label, args):
    prediction_knn = weighted_knn(feature_bank, feature_bank, modified_label, args.num_classes, args.k, 10)  # temperature in weighted KNN
    vote_y = torch.gather(prediction_knn, 1, modified_label.view(-1, 1)).squeeze()
    vote_max = prediction_knn.max(dim=1)[0]
    right_score = vote_y / vote_max
    clean_candidate_ids = torch.where(right_score >= args.theta_s)[0]
    undecided_ids = torch.where(right_score < args.theta_s)[0]
    return clean_candidate_ids, undecided_ids, torch.tensor([], dtype=torch.int64).cuda(), torch.tensor([], dtype=torch.int64).cuda()


def select_samples_extended_excluded(feature_bank, modified_label, args, stability_score):
    prediction_knn = weighted_knn(feature_bank, feature_bank, modified_label, args.num_classes, args.k, 10)  # temperature in weighted KNN
    vote_y = torch.gather(prediction_knn, 1, modified_label.view(-1, 1)).squeeze()
    vote_max = prediction_knn.max(dim=1)[0]
    right_score = vote_y / vote_max
    clean_candidate_ids = torch.where(right_score >= args.theta_s)[0]
    undecided_ids = torch.where(right_score < args.theta_s)[0]

    clean_id_extended = torch.tensor([], dtype=torch.int64).cuda()
    clean_id_excluded = torch.tensor([], dtype=torch.int64).cuda()
    if stability_score.size()[0] > 0:
        clean_id_extended = torch.where(stability_score >= (args.theta_extend * torch.max(stability_score).detach().item()))[0]
        clean_id_excluded = torch.where(stability_score <= (args.theta_exclude))[0]

    return clean_candidate_ids, undecided_ids, clean_id_extended, clean_id_excluded

def select_samples_stable(feature_bank, modified_label, args, stability_score):
    prediction_knn = weighted_knn(feature_bank, feature_bank, modified_label, args.num_classes, args.k, 10)  # temperature in weighted KNN
    vote_y = torch.gather(prediction_knn, 1, modified_label.view(-1, 1)).squeeze()
    vote_max = prediction_knn.max(dim=1)[0]
    right_score = vote_y / vote_max
    clean_candidate_ids = torch.where(right_score >= args.theta_s)[0]
    undecided_ids = torch.where(right_score < args.theta_s)[0]

    clean_id_extended = torch.tensor([], dtype=torch.int64).cuda()
    clean_id_excluded = torch.tensor([], dtype=torch.int64).cuda()

    return clean_candidate_ids, undecided_ids, clean_id_extended, clean_id_excluded
