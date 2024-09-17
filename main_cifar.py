import argparse
import pickle

import torch.optim.lr_scheduler
import torchvision.transforms as transforms
import wandb
from torch.optim import SGD
from torch.utils.data import Subset
from tqdm import tqdm
from utils.feature import extract_cifar_features as extract_features
from utils.relabeling import relabel_samples
from utils.sample_selection import select_samples

from datasets.dataloader_cifar import cifar_dataset
from models.preresnet import PreResNet18
from utils import *

parser = argparse.ArgumentParser('Train with synthetic cifar noisy dataset')

parser.add_argument('--project', default="cifar10_test", type=str, help='project name')
parser.add_argument('--dataset_path', default='CIFAR/CIFAR10', help='dataset path')
parser.add_argument('--noisy_dataset_path', default='CIFAR/CIFAR100', help='open-set noise dataset path')
parser.add_argument('--dataset', default='cifar10', help='dataset name')
parser.add_argument('--noisy_dataset', default='cifar100', help='open-set noise dataset name')

# dataset settings
parser.add_argument('--noise_mode', default='sym', type=str, help='artifical noise mode (default: symmetric)')
parser.add_argument('--noise_ratio', default=0.5, type=float, help='artifical noise ratio (default: 0.5)')
parser.add_argument('--open_ratio', default=0.0, type=float, help='artifical noise ratio (default: 0.0)')

# model settings
parser.add_argument('--theta_s', default=1.0, type=float, help='threshold for selecting samples (default: 1)')
parser.add_argument('--theta_r', default=0.9, type=float, help='threshold for relabelling samples (default: 0.9)')
parser.add_argument('--lambda_fc', default=1.0, type=float, help='weight of feature consistency loss (default: 1.0)')
parser.add_argument('--k', default=200, type=int, help='neighbors for knn sample selection (default: 200)')

# train settings
parser.add_argument('--model', default='PreResNet18', help=f'model architecture (default: PreResNet18)')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run (default: 300)')
parser.add_argument('--relabeling_strategy', default='base_line', type=str, help='relabeling_strategy')
parser.add_argument('--sampling_strategy', default='base_line', type=str, help='sampling_strategy')

parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate (default: 0.02)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver (default: 0.9)')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--seed', default=3047, type=int, help='seed for initializing training. (default: 3047)')
parser.add_argument('--gpuid', default='0', type=str, help='Selected GPU (default: "0")')
parser.add_argument('--entity', type=str, help='Wandb user entity')
parser.add_argument('--run_path', type=str, help='run path containing all results')


def train(labeled_trainloader, modified_label, all_trainloader, encoder, classifier, proj_head, pred_head, optimizer, epoch, args):
    encoder.train()
    classifier.train()
    proj_head.train()
    pred_head.train()
    xlosses = AverageMeter('xloss')
    ulosses = AverageMeter('uloss')
    labeled_train_iter = iter(labeled_trainloader)
    all_bar = tqdm(all_trainloader)
    for batch_idx, ([inputs_u1, inputs_u2], _, _, _) in enumerate(all_bar):
        try:
            # [inputs_x1, inputs_x2], labels_x, _, index = labeled_train_iter.next()
            [inputs_x1, inputs_x2], labels_x, _, index = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            # [inputs_x1, inputs_x2], labels_x, _, index = labeled_train_iter.next()
            [inputs_x1, inputs_x2], labels_x, _, index = next(labeled_train_iter)

        # cross-entropy training with mixup
        batch_size = inputs_x1.size(0)
        inputs_x1, inputs_x2 = inputs_x1.cuda(), inputs_x2.cuda()
        labels_x = modified_label[index]
        targets_x = torch.zeros(batch_size, args.num_classes, device=inputs_x1.device).scatter_(1, labels_x.view(-1, 1), 1)
        l = np.random.beta(4, 4)
        l = max(l, 1 - l)
        all_inputs_x = torch.cat([inputs_x1, inputs_x2], dim=0)
        all_targets_x = torch.cat([targets_x, targets_x], dim=0)
        idx = torch.randperm(all_inputs_x.size()[0])
        input_a, input_b = all_inputs_x, all_inputs_x[idx]
        target_a, target_b = all_targets_x, all_targets_x[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = classifier(encoder(mixed_input))
        Lce = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))

        # optional feature-consistency
        inputs_u1, inputs_u2 = inputs_u1.cuda(), inputs_u2.cuda()

        feats_u1 = encoder(inputs_u1)
        feats_u2 = encoder(inputs_u2)
        f, h = proj_head, pred_head

        z1, z2 = f(feats_u1), f(feats_u2)
        p1, p2 = h(z1), h(z2)
        Lfc = D(p2, z1)
        loss = Lce + args.lambda_fc * Lfc
        xlosses.update(Lce.item())
        ulosses.update(Lfc.item())
        all_bar.set_description(
            f'Train epoch {epoch} LR:{optimizer.param_groups[0]["lr"]} Labeled loss: {xlosses.avg:.4f} Unlabeled loss: {ulosses.avg:.4f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logger.log({'ce loss': xlosses.avg, 'fc loss': ulosses.avg})


def test(testloader, encoder, classifier, epoch):
    encoder.eval()
    classifier.eval()
    accuracy = AverageMeter('accuracy')
    data_bar = tqdm(testloader)
    with torch.no_grad():
        for i, (data, label, _) in enumerate(data_bar):
            data, label = data.cuda(), label.cuda()
            feat = encoder(data)
            res = classifier(feat)
            pred = torch.argmax(res, dim=1)
            acc = torch.sum(pred == label) / float(data.size(0))
            accuracy.update(acc.item(), data.size(0))
            data_bar.set_description(f'Test epoch {epoch}: Accuracy#{accuracy.avg:.4f}')
    return accuracy.avg


def evaluate(dataloader, encoder, classifier, args, noisy_label):
    encoder.eval()
    classifier.eval()

    ################################### feature extraction ###################################
    with torch.no_grad():
        feature_bank, preds_logits = extract_features(dataloader, encoder, classifier)

        relabeling_result = relabel_samples(torch.cat(preds_logits, dim=0), noisy_label, args)
        relabeled_ids, modified_label, modified_score, noisy_labels_score = relabeling_result

        clean_candidate_ids, undecided_ids = select_samples(feature_bank, modified_label, args)

    return clean_candidate_ids, undecided_ids, modified_label, modified_score, relabeled_ids, noisy_labels_score


def main():
    args = parser.parse_args()
    seed_everything(args.seed)
    if args.run_path is None:
        args.run_path = f'Dataset({args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode})_Model({args.theta_r}_{args.theta_s})'

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    global logger
    logger = wandb.init(project=args.dataset, entity=args.entity, name=args.run_path, group=args.dataset)
    logger.config.update(args)

    # generate noisy dataset with our transformation
    if not os.path.isdir(f'{args.dataset}'):
        os.mkdir(f'{args.dataset}')
    if not os.path.isdir(f'{args.dataset}/{args.run_path}'):
        os.mkdir(f'{args.dataset}/{args.run_path}')

    ############################# Dataset initialization ##############################################
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.image_size = 32
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.image_size = 32
        normalize = transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
    else:
        raise ValueError(f'args.dataset should be cifar10 or cifar100, rather than {args.dataset}!')

    # data loading
    weak_transform = transforms.Compose([
        transforms.RandomCrop(args.image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    none_transform = transforms.Compose([transforms.ToTensor(), normalize])
    strong_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           CIFAR10Policy(),
                                           transforms.ToTensor(),
                                           normalize])

    # generate train dataset with only filtered clean subset
    train_data = cifar_dataset(dataset=args.dataset, root_dir=args.dataset_path,
                               noise_data_dir=args.noisy_dataset_path, noisy_dataset=args.noisy_dataset,
                               transform=KCropsTransform(strong_transform, 2), open_ratio=args.open_ratio,
                               dataset_mode='train', noise_ratio=args.noise_ratio, noise_mode=args.noise_mode,
                               noise_file=f'{args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode}_noise.json')
    eval_data = cifar_dataset(dataset=args.dataset, root_dir=args.dataset_path, transform=weak_transform,
                              noise_data_dir=args.noisy_dataset_path, noisy_dataset=args.noisy_dataset,
                              dataset_mode='train', noise_ratio=args.noise_ratio, noise_mode=args.noise_mode,
                              open_ratio=args.open_ratio,
                              noise_file=f'{args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode}_noise.json')
    test_data = cifar_dataset(dataset=args.dataset, root_dir=args.dataset_path, transform=none_transform,
                              noise_data_dir=args.noisy_dataset_path, noisy_dataset=args.noisy_dataset,
                              dataset_mode='test')
    all_data = cifar_dataset(dataset=args.dataset, root_dir=args.dataset_path,
                                   noise_data_dir=args.noisy_dataset_path, noisy_dataset=args.noisy_dataset,
                                   transform=MixTransform(strong_transform=strong_transform, weak_transform=weak_transform, K=1),
                                   open_ratio=args.open_ratio,
                                   dataset_mode='train', noise_ratio=args.noise_ratio, noise_mode=args.noise_mode,
                                   noise_file=f'{args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode}_noise.json')

    # extract noisy labels and clean labels for performance monitoring
    noisy_label = torch.tensor(eval_data.cifar_label).cuda()
    clean_label = torch.tensor(eval_data.clean_label).cuda()

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_loader = torch.utils.data.DataLoader(all_data, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

    ################################ Model initialization ###########################################
    encoder = PreResNet18(args.num_classes)
    classifier = torch.nn.Linear(encoder.fc.in_features, args.num_classes)
    proj_head = torch.nn.Sequential(torch.nn.Linear(encoder.fc.in_features, 256),
                                    torch.nn.BatchNorm1d(256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 128))
    pred_head = torch.nn.Sequential(torch.nn.Linear(128, 256),
                                    torch.nn.BatchNorm1d(256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 128))
    encoder.fc = torch.nn.Identity()

    encoder.cuda()
    classifier.cuda()
    proj_head.cuda()
    pred_head.cuda()

    #################################### Training initialization #######################################
    optimizer = SGD([{'params': encoder.parameters()}, {'params': classifier.parameters()}, {'params': proj_head.parameters()}, {'params': pred_head.parameters()}],
                    lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/50.0)

    acc_logs = open(f'{args.dataset}/{args.run_path}/acc.txt', 'w')
    stat_logs = open(f'{args.dataset}/{args.run_path}/stat.txt', 'w')
    save_config(args, f'{args.dataset}/{args.run_path}')
    print('Train args: \n', args)
    best_acc = 0

    clean_candidate_ids_per_epochs = {}
    relabeled_ids_per_epochs = {}

    ################################ Training loop ###########################################
    for i in range(args.epochs):
        # clean_candidate_ids, undecided_ids, modified_label = evaluate(eval_loader, encoder, classifier, args, noisy_label, clean_label, i, stat_logs)
        evaluate_result = evaluate(eval_loader, encoder, classifier, args, noisy_label)
        clean_candidate_ids, undecided_ids, modified_label, modified_score, relabeled_ids, noisy_labels_score = evaluate_result

        TP = torch.sum(modified_label[clean_candidate_ids] == clean_label[clean_candidate_ids])
        FP = torch.sum(modified_label[clean_candidate_ids] != clean_label[clean_candidate_ids])
        TN = torch.sum(modified_label[undecided_ids] != clean_label[undecided_ids])
        FN = torch.sum(modified_label[undecided_ids] == clean_label[undecided_ids])
        # print(f'Epoch [{i}/{args.epochs}] selection: theta_s:{args.theta_s} TP: {TP} FP:{FP} TN:{TN} FN:{FN}')
        correct = torch.sum(modified_label[relabeled_ids] == clean_label[relabeled_ids])
        all = len(relabeled_ids)
        precision = (TP)/(TP + FP)
        recall = (TP)/(TP + FN)
        f1 = (2 * precision * recall) / (precision + recall)

        clean_candidate_ids_per_epochs[i] = clean_candidate_ids.detach().cpu()
        relabeled_ids_per_epochs[i] = relabeled_ids.detach().cpu()
        logger.log({'epoch': i})
        logger.log({
            'TP': TP,'FP': FP, 'TN': TN, 'FN': FN,
            "Precision": precision, "Recall": recall, "F1": f1,

            'number of clean candidate ids': clean_candidate_ids.size()[0],
            'number of undecided ids': undecided_ids.size()[0],
            'number of relabeled ids': relabeled_ids.size()[0],
            'number of correctly relabeled ids': correct,

            'modified score min': torch.min(modified_score),
            'modified score mean': torch.mean(modified_score),
            'modified score max': torch.max(modified_score),

            'noisy labels score min': torch.min(noisy_labels_score),
            'noisy labels score mean': torch.mean(noisy_labels_score),
            'noisy labels score max': torch.max(noisy_labels_score),
            
            'relabeld samples noisy label score min': 0 if noisy_labels_score[relabeled_ids].size()[0] == 0 else torch.min(noisy_labels_score[relabeled_ids]),
            'relabeld samples noisy label score mean': 0 if noisy_labels_score[relabeled_ids].size()[0] == 0 else torch.mean(noisy_labels_score[relabeled_ids]),
            'relabeld samples noisy label score max': 0 if noisy_labels_score[relabeled_ids].size()[0] == 0 else torch.max(noisy_labels_score[relabeled_ids]),

            'relabeld samples relabeld label score min': 0 if modified_score[relabeled_ids].size()[0] == 0 else torch.min(modified_score[relabeled_ids]),
            'relabeld samples relabeld label score mean': 0 if modified_score[relabeled_ids].size()[0] == 0 else torch.mean(modified_score[relabeled_ids]),
            'relabeld samples relabeld label score max': 0 if modified_score[relabeled_ids].size()[0] == 0 else torch.max(modified_score[relabeled_ids]),
        })

        # print(f'Epoch [{i}/{args.epochs}] relabelling:  correct: {correct} original: {orginal} total: {all}')

        stat_logs.write(f'Epoch [{i}/{args.epochs}] selection: theta_s:{args.theta_s} TP: {TP} FP:{FP} TN:{TN} FN:{FN}\n')
        stat_logs.write(f'Epoch [{i}/{args.epochs}] relabelling:  correct: {correct} total: {all}\n')
        stat_logs.flush()


        # balanced_sampler
        clean_subset = Subset(train_data, clean_candidate_ids.cpu())
        sampler = ClassBalancedSampler(labels=modified_label[clean_candidate_ids], num_classes=args.num_classes)
        labeled_loader = torch.utils.data.DataLoader(clean_subset, batch_size=args.batch_size, sampler=sampler, num_workers=4, drop_last=True)
     
        train(labeled_loader, modified_label, all_loader, encoder, classifier, proj_head, pred_head, optimizer, i, args)

        cur_acc = test(test_loader, encoder, classifier, i)
        logger.log({'acc': cur_acc, 'best acc': best_acc})
        scheduler.step()
        if cur_acc > best_acc:
            best_acc = cur_acc
            save_checkpoint({
                'cur_epoch': i,
                'classifier': classifier.state_dict(),
                'encoder': encoder.state_dict(),
                'proj_head': proj_head.state_dict(),
                'pred_head': pred_head.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=f'{args.dataset}/{args.run_path}/best_acc.pth.tar')
        acc_logs.write(f'Epoch [{i}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')
        acc_logs.flush()
        print(f'Epoch [{i}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')

    with open(f'logs/cifar10_{args.noise_mode}_{args.noise_ratio}_clean_candidate_ids_per_epochs.pkl', 'wb') as f:
        pickle.dump(clean_candidate_ids_per_epochs, f)

    with open(f'logs/cifar10_{args.noise_mode}_{args.noise_ratio}_relabeled_ids_per_epochs.pkl', 'wb') as f:
        pickle.dump(relabeled_ids_per_epochs, f)

    save_checkpoint({
        'cur_epoch': args.epochs,
        'classifier': classifier.state_dict(),
        'encoder': encoder.state_dict(),
        'proj_head': proj_head.state_dict(),
        'pred_head': pred_head.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=f'{args.dataset}/{args.run_path}/last.pth.tar')


if __name__ == '__main__':
    main()
