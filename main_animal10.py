import argparse

import torch.optim.lr_scheduler
import torchvision.transforms as transforms
import wandb
import pickle
from torch.optim import SGD
from torch.utils.data import Subset
from tqdm import tqdm
import torch.nn as nn

from datasets.dataloader_animal10n import animal_dataset
from torchvision.models import vgg19_bn
from utils import *
from utils.relabeling import relabel_samples
from utils.sample_selection import select_samples
from utils.feature import extract_features

parser = argparse.ArgumentParser('Train with ANIMAL-10N dataset')
parser.add_argument('--project', default="animal10n_test", type=str, help='project name')
parser.add_argument('--dataset_path', default='ANIMAL-10N', help='dataset path')

# model settings
parser.add_argument('--lambda_fc', default=1.0, type=float, help='weight of feature consistency loss (default: 1.0)')
parser.add_argument('--theta_s', default=1.0, type=float, help='threshold for voted correct samples (default: 1.0)')
parser.add_argument('--theta_r', default=0.95, type=float, help='threshold for relabel samples (default: 0.95)')
parser.add_argument('--k', default=200, type=int, help='neighbors for soft-voting (default: 200)')

# train settings
parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run (default: 200)')
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
    for batch_idx, ([inputs_u1, inputs_u2],  _, _) in enumerate(all_bar):
        try:
            [inputs_x1, inputs_x2], labels_x, index = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            [inputs_x1, inputs_x2], labels_x, index = next(labeled_train_iter)

        # cross-entropy training with mixup
        batch_size = inputs_x1.size(0)
        inputs_x1, inputs_x2 = inputs_x1.cuda(), inputs_x2.cuda()
        labels_x = modified_label[index]
        targets_x = torch.zeros(batch_size, args.num_classes, device=inputs_x1.device).scatter_(1, labels_x.view(-1, 1), 1)
        l = np.random.beta(0.5, 0.5)
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
            batch_pred_logits = classifier(feat)
            pred = torch.argmax(batch_pred_logits, dim=1)
            acc = torch.sum(pred == label) / float(data.size(0))
            accuracy.update(acc.item(), data.size(0))
            data_bar.set_description(f'Test epoch {epoch}: Accuracy#{accuracy.avg:.4f}')
    return accuracy.avg


def evaluate(dataloader, encoder, classifier, args, human_labels):
    encoder.eval()
    classifier.eval()

    with torch.no_grad():
        feature_bank, preds_logits = extract_features(dataloader, encoder, classifier)

        relabeling_result = relabel_samples(torch.cat(preds_logits, dim=0), human_labels, args)
        relabeled_ids, modified_label, modified_score, human_labels_score = relabeling_result

        clean_candidate_ids, undecided_ids = select_samples(feature_bank, modified_label, args)
    return clean_candidate_ids, undecided_ids, modified_label, modified_score, relabeled_ids, human_labels_score


def main():
    args = parser.parse_args()
    seed_everything(args.seed)
    if args.run_path is None:
        args.run_path = f'Dataset(animal10n_Model({args.theta_r}_{args.theta_s})'

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    global logger
    logger = wandb.init(project=args.project, entity=args.entity, name=args.run_path)
    logger.config.update(args)

    if not os.path.isdir(f'animal10n'):
        os.mkdir(f'animal10n')
    if not os.path.isdir(f'animal10n/{args.run_path}'):
        os.mkdir(f'animal10n/{args.run_path}')

    ############################# Dataset initialization ##############################################
    args.num_classes = 10
    args.image_size = 64

    # data loading
    weak_transform = transforms.Compose([
        transforms.RandomCrop(args.image_size, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    none_transform = transforms.Compose([transforms.ToTensor()])  # no augmentation
    strong_transform = transforms.Compose([transforms.RandomCrop(args.image_size, padding=8),
                                           transforms.RandomHorizontalFlip(),
                                           RandAugment(),
                                           transforms.ToTensor()])

    # eval data served as soft-voting pool
    train_data = animal_dataset(root=args.dataset_path, transform=KCropsTransform(strong_transform, 2), mode='train')
    eval_data = animal_dataset(root=args.dataset_path, transform=weak_transform, mode='train')
    test_data = animal_dataset(root=args.dataset_path, transform=none_transform, mode='test')
    all_data = animal_dataset(root=args.dataset_path, transform=MixTransform(strong_transform, weak_transform, 1), mode='train')

    # noisy labels
    human_labels = torch.tensor(eval_data.targets).cuda()

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_loader = torch.utils.data.DataLoader(all_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    ################################ Model initialization ###########################################
    encoder = vgg19_bn(pretrained=False)
    encoder.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(p=0.2),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(p=0.2),
    )

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, 0.005)
            nn.init.constant_(m.bias, 0)

    encoder.classifier.apply(init_weights)
    classifier = nn.Linear(4096, args.num_classes)
    proj_head = torch.nn.Sequential(torch.nn.Linear(4096, 256),
                                    torch.nn.BatchNorm1d(256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 128))
    pred_head = torch.nn.Sequential(torch.nn.Linear(128, 256),
                                    torch.nn.BatchNorm1d(256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 128))
    classifier.apply(init_weights)

    encoder.cuda()
    classifier.cuda()
    proj_head.cuda()
    pred_head.cuda()

    #################################### Training initialization #######################################
    optimizer = SGD([{'params': encoder.parameters()}, {'params': classifier.parameters()}, {'params': proj_head.parameters()}, {'params': pred_head.parameters()}],
                    lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

    acc_logs = open(f'animal10n/{args.run_path}/acc.txt', 'w')
    save_config(args, f'animal10n/{args.run_path}')
    print('Train args: \n', args)
    best_acc = 0

    clean_candidate_ids_per_epochs = {}
    relabeled_ids_per_epochs = {}

    ################################ Training loop ###########################################
    for i in range(args.epochs):
        evaluate_result = evaluate(eval_loader, encoder, classifier, args, human_labels)
        clean_candidate_ids, undecided_ids, modified_label, modified_score, relabeled_ids, human_labels_score = evaluate_result
        clean_candidate_ids_per_epochs[i] = clean_candidate_ids.detach().cpu()
        relabeled_ids_per_epochs[i] = relabeled_ids.detach().cpu()
        logger.log({'epoch': i})
        logger.log({
            'number of clean candidate ids': clean_candidate_ids.size()[0],
            'number of undecided ids': undecided_ids.size()[0],
            'number of relabeled ids': relabeled_ids.size()[0],

            'modified score min': torch.min(modified_score),
            'modified score mean': torch.mean(modified_score),
            'modified score max': torch.max(modified_score),

            'human labels score min': torch.min(human_labels_score),
            'human labels score mean': torch.mean(human_labels_score),
            'human labels score max': torch.max(human_labels_score),
            
            'relabeld samples human label score min': 0 if human_labels_score[relabeled_ids].size()[0] == 0 else torch.min(human_labels_score[relabeled_ids]),
            'relabeld samples human label score mean': 0 if human_labels_score[relabeled_ids].size()[0] == 0 else torch.mean(human_labels_score[relabeled_ids]),
            'relabeld samples human label score max': 0 if human_labels_score[relabeled_ids].size()[0] == 0 else torch.max(human_labels_score[relabeled_ids]),

            'relabeld samples relabeld label score min': 0 if modified_score[relabeled_ids].size()[0] == 0 else torch.min(modified_score[relabeled_ids]),
            'relabeld samples relabeld label score mean': 0 if modified_score[relabeled_ids].size()[0] == 0 else torch.mean(modified_score[relabeled_ids]),
            'relabeld samples relabeld label score max': 0 if modified_score[relabeled_ids].size()[0] == 0 else torch.max(modified_score[relabeled_ids]),
        })
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
            }, filename=f'animal10n/{args.run_path}/best_acc.pth.tar')
        acc_logs.write(f'Epoch [{i}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')
        acc_logs.flush()
        print(f'Epoch [{i}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')

    with open('logs/animal10_clean_candidate_ids_per_epochs.pkl', 'wb') as f:
        pickle.dump(clean_candidate_ids_per_epochs, f)

    with open('logs/relabeled_ids_per_epochs.pkl', 'wb') as f:
        pickle.dump(relabeled_ids_per_epochs, f)

    save_checkpoint({
        'cur_epoch': i,
        'classifier': classifier.state_dict(),
        'encoder': encoder.state_dict(),
        'proj_head': proj_head.state_dict(),
        'pred_head': pred_head.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=f'animal10n/{args.run_path}/last.pth.tar')


if __name__ == '__main__':
    main()
