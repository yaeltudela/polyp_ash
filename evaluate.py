import time

import torch
from torch.utils.data import DataLoader

from data.data_loader import build_dataset
from losses import dice_score
from metrics import iou_score, compute_cls_metrics
from model.model_builder import build_models
from train import estimate_cls_probs
from utils.io import build_name
from utils.utils import parse_args, resize


@torch.no_grad()
def eval_model(data_loader, model, num_classes, save_splits=None):
    # embeds = []
    # def get_embeds(module, inputs, outputs):
    #     embeds.append(outputs.detach().cpu())
    #
    # model.backbone.encoder.norm.register_forward_hook(get_embeds)
    total_time = 0
    if save_splits is None:
        save_splits = ['train', 'valid', 'val', 'test']
    else:
        if isinstance(save_splits, str):
            save_splits = [save_splits]
    split = data_loader.dataset.split
    targets_list, preds_list, annot_segm_list, output_segm_list = [], [], [], []

    model.eval()

    end = time.time()
    for idx, batch in enumerate(data_loader):
        samples, targets, annots = batch
        # images, annots = batch

        samples = samples.cuda()
        targets = targets.cuda()
        annots = annots.cuda().float()

        cls_outputs, outputs = model(samples)

        if isinstance(outputs, list):
            outputs = outputs[0]

        # if model down-sample the results, resize it properly
        if outputs.size(-1) != annots.size(-1):
            annots = resize(annots, outputs.shape[2:])

        cls_outputs = estimate_cls_probs(outputs)

        # measure elapsed time
        total_time += (time.time() - end)
        end = time.time()

        targets_list.append(targets.cpu())
        preds_list.append(cls_outputs.cpu())
        output_segm_list.append(outputs.detach().cpu())
        annot_segm_list.append(annots.cpu())

        # if split in save_splits:
        #     save_model_results(samples, annots, outputs, targets, f'visualization/{split}/{idx}.jpg')

    preds_list = torch.cat(preds_list, 0)  # b, N
    targets_list = torch.cat(targets_list, 0)  # b, 1

    output_segm_list = torch.cat(output_segm_list, 0)
    annot_segm_list = torch.cat(annot_segm_list, 0)

    all_ious = iou_score(output_segm_list, annot_segm_list, reduction=None)

    # remove bg class for cls metrics purpose and binarize preds
    targets_list -= 1
    preds_list = preds_list[:, 1:].argmax(1)

    cf, f1, f2, mcc, npv, precision, recall, results, spec = compute_cls_metrics(None, None, preds_list, targets_list,
                                                                                 num_classes - 1)
    # binarize and recompute for comparison
    bin_output_segm_list = output_segm_list.argmax(1, keepdim=True).bool().long()
    bin_annot_segm_list = annot_segm_list.argmax(1, keepdim=True).bool().long()
    bin_output_segm_list = torch.nn.functional.one_hot(bin_output_segm_list).squeeze(1).permute(0, 3, 1, 2)
    bin_annot_segm_list = torch.nn.functional.one_hot(bin_annot_segm_list).squeeze(1).permute(0, 3, 1, 2)

    bin_iou = iou_score(bin_output_segm_list, bin_annot_segm_list, None)
    bin_dice = dice_score(bin_output_segm_list, bin_annot_segm_list)

    results.append(f'Mean time: {total_time}')
    results.append(f"dataset / time: {len(data_loader.dataset) / total_time}")
    results.append(f'MIoU: {all_ious.mean():.4f}')
    results.append(f'IoU per class: {all_ious}')
    results.append(f'MIoU (bg / fg): {bin_iou.mean():.4f}')
    results.append(f'IoU per class (bg / fg): {bin_iou}')
    results.append(f'Dice score bin: {bin_dice.item()}')

    for result in results:
        print(result)

    return results


if __name__ == '__main__':

    args = parse_args()

    if not args.output.startswith("results"):
        output_folder = build_name(args, True)
    else:
        output_folder = args.output

    im_size = 224
    device = 'cuda'
    use_fpn = args.use_fpn
    model_config = args.config

    batch_size = 8
    extra_losses = args.extra_losses
    num_workers = 4
    decoder_name = args.decoder_name
    head_name = args.head_name

    model = build_models(224, use_fpn, decoder_name, head_name, model_config, 2, extra_losses, verbose=True)
    model.to('cuda')

    try:
        weights = torch.load(f"{output_folder}/model.pth")
        model.load_state_dict(weights)
    except:
        raise AssertionError("fail in loading weights")

    datasets = [
        'test_CLINIC', 'test_COLON', 'test_ETIS', 'test_KVASIR']
    for dataset_name in datasets:
        _, test_ds, _, _ = build_dataset(dataset_name, im_size, use_ass=True, only_polyp=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=True)

        print("-------------------------------------------------")
        print(test_loader.dataset.dataset_name)
        print("-------------------------------------------------")

        eval_model(test_loader, model, 4, save_splits=('test',))
