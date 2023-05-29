import datetime
import time

import torch
import wandb
from timm.utils import accuracy, AverageMeter
from torch.cuda import amp
from tqdm import tqdm

from criterion import Criterion
from data.data_loader import get_train_test_loaders
from metrics import compute_cls_metrics, iou_score
from model.model_builder import build_models
from optim import build_scheduler, build_optimizer
from scaler import NativeScalerWithGradNormCount
from utils.io import save_config, save_model, save_results, build_name
from utils.utils import EarlyStopping, setup, is_debug, resize
from utils.visualization import save_model_results


def train(model, optimizer, scaler, lr_scheduler, criterion, loader, epochs, accum_iter, val_loader=None,
          mixup_fn=None, use_amp=True, num_classes=None, early_stop=None, log_to_wandb=False, val_criterion=None):

    bar = tqdm(range(epochs))
    stop = False
    for epoch in bar:

        losses_dict = train_one_epoch(model, criterion, loader, optimizer, epoch, lr_scheduler, accum_steps=accum_iter,
                                      loss_scaler=scaler, mixup_fn=mixup_fn, use_amp=use_amp)

        if val_loader is not None:
            _, val_loss, val_miou, _ = eval_model(val_loader, model, val_criterion, num_classes,
                                                  save_splits='valid' if epoch % 10 == 0 else '')
            losses_dict.update({
                'val_loss': val_loss,
                'val_miou': val_miou,
            })
        else:
            # do not use early stopping, log train loss instead
            val_loss = losses_dict['train_loss']

        if early_stop is not None:
            stop = early_stop(val_loss, model)

            losses_dict['tol'] = early_stop.tol
            losses_dict['min_loss'] = early_stop.best

        bar.set_postfix(losses_dict)

        if log_to_wandb:
            wandb.log(losses_dict)

        #  checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'o_model': optimizer.state_dict(),
            }, "checkpoint.pth")

        if stop:
            if log_to_wandb:
                wandb.log({'stop_epoch': epoch})
            break

    return model


def train_one_epoch(model, criterion, data_loader, optimizer, epoch, lr_scheduler, accum_steps=1, print_freq=100,
                    loss_scaler=None, mixup_fn=None, use_amp=True):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    miou_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, batch in enumerate(data_loader):
        samples, targets, annots = batch
        empty_annots = annots[:, 1:].sum((1, 2, 3)) == 0
        targets[empty_annots] = 0

        samples = samples.cuda()
        targets = targets.cuda()
        annots = annots.cuda().float()

        if mixup_fn is not None:
            samples, targets, annots = mixup_fn(samples, targets, annots)

        # show_batch(samples, targets, annots)
        with amp.autocast(enabled=use_amp):
            cls_outputs, outputs = model(samples)

        outputs_aux = {}
        if isinstance(outputs, list):
            outputs, outputs_aux = outputs[0], outputs[1:]

            outputs_aux = {f'base_aux{i.size(-1)}': {'preds': resize(i, annots.shape[2:]), 'targets': annots,
                                                     'scale_factor': i.size(-1) / annots.size(-1)} for i in outputs_aux}

        # if model down-sample the results, resize it properly
        if outputs.size(-1) != annots.size(-1):
            outputs = resize(outputs, annots.shape[2:])

        loss_dict = {
            'base': {
                'preds': outputs,
                'targets': annots
            },
            'cls': {
                'preds': cls_outputs,
                'targets': targets
            },
            'iou': {
                'preds': outputs,
                'targets': annots
            },
        }

        loss_dict.update(outputs_aux)
        loss = criterion(loss_dict)
        loss = loss / accum_steps

        grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(idx + 1) % accum_steps == 0)

        if (idx + 1) % accum_steps == 0:
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step_update((epoch * num_steps + idx) // accum_steps)
        loss_scale_value = loss_scaler.state_dict()
        if loss_scale_value:
            loss_scale_value = loss_scale_value["scale"]
            scaler_meter.update(loss_scale_value)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), annots.size(0))
        miou = iou_score(outputs, annots)
        miou_meter.update(miou.item(), annots.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if epoch % 10 == 0:
            save_model_results(samples, annots, outputs, f'visualization/train/{idx}.jpg')
        if idx % print_freq == 0 or idx == len(data_loader):
            lr = optimizer.param_groups[-1]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            print(
                f"Train epoch: [{epoch}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB")

    epoch_time = time.time() - start
    print(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return {'train_loss': loss_meter.avg, 'miou_train': miou_meter.avg, 'grad_norm': norm_meter.avg}


def estimate_cls_probs(predictions, threshold=0.0):
    b, classes = predictions.size(0), predictions.size(1)
    out = torch.zeros(b, classes, dtype=torch.long, device=predictions.device)

    if threshold is not None:
        predictions = predictions.sigmoid()
        predictions[predictions < threshold] = 0.
    predictions = predictions.argmax(1).flatten(1)
    ones = torch.ones_like(predictions)

    out = out.scatter_add(1, predictions, ones)

    probs = out / (out.sum(1).view(-1, 1) + 1e-10)

    return probs


@torch.no_grad()
def eval_model(data_loader, model, criterion, num_classes, save_splits=None, wandb_log=False):
    # embeds = []
    # def get_embeds(module, inputs, outputs):
    #     embeds.append(outputs.detach().cpu())
    #
    # model.backbone.encoder.norm.register_forward_hook(get_embeds)

    if save_splits is None:
        save_splits = ['train', 'valid', 'val', 'test']
    else:
        if isinstance(save_splits, str):
            save_splits = [save_splits]
    split = data_loader.dataset.split
    targets_list, preds_list, annot_segm_list, output_segm_list = [], [], [], []

    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    miou_meter = AverageMeter()

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
        loss_dict = {
            'base': {
                'preds': outputs,
                'targets': annots
            },
            'cls': {
                'preds': cls_outputs,
                'targets': targets
            },
            'iou': {
                'preds': outputs,
                'targets': annots
            },
            'cons': {
                'preds': cls_outputs,
                'targets': targets
            }
        }
        loss = criterion(loss_dict)

        # compute cls accuracy by removing bg logit and restoring targets ids
        acc1, = accuracy(cls_outputs[:, 1:], (targets - 1), topk=(1,))
        miou = iou_score(outputs, annots)

        loss_meter.update(loss.item(), samples.size(0))
        acc1_meter.update(acc1.item(), samples.size(0))
        miou_meter.update(miou.item(), samples.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        targets_list.append(targets.cpu())
        preds_list.append(cls_outputs.cpu())
        output_segm_list.append(outputs.detach().cpu())
        annot_segm_list.append(annots.cpu())

        if split in save_splits:
            save_model_results(samples, annots, outputs, f'visualization/{split}/{idx}.jpg')

    print(f' * Acc@1 {acc1_meter.avg:.3f} mIoU {miou_meter.avg:.3f}')

    results = None
    if data_loader.dataset.split == 'test':
        preds_list = torch.cat(preds_list, 0)  # b, N
        targets_list = torch.cat(targets_list, 0)  # b, 1

        output_segm_list = torch.cat(output_segm_list, 0)
        annot_segm_list = torch.cat(annot_segm_list, 0)

        all_ious = iou_score(output_segm_list, annot_segm_list, reduction=None)

        # remove bg class for cls metrics purpose and binarize preds
        targets_list -= 1
        preds_list = preds_list[:, 1:].argmax(1)

        cf, f1, f2, mcc, npv, precision, recall, results, spec = compute_cls_metrics(acc1_meter, loss_meter,
                                                                                     preds_list, targets_list,
                                                                                     num_classes - 1)
        # binarize and recompute for comparison
        bin_output_segm_list = output_segm_list.argmax(1, keepdim=True).bool().long()
        bin_annot_segm_list = annot_segm_list.argmax(1, keepdim=True).bool().long()
        bin_output_segm_list = torch.nn.functional.one_hot(bin_output_segm_list).squeeze(1).permute(0, 3, 1, 2)
        bin_annot_segm_list = torch.nn.functional.one_hot(bin_annot_segm_list).squeeze(1).permute(0, 3, 1, 2)

        bin_iou = iou_score(bin_output_segm_list, bin_annot_segm_list, None)

        results.append(f'MIoU: {all_ious.mean():.4f}')
        results.append(f'IoU per class: {all_ious}')
        results.append(f'MIoU (bg / fg): {bin_iou.mean():.4f}')
        results.append(f'IoU per class (bg / fg): {bin_iou}')

        if wandb_log:
            print('logging results to wandb')
            wandb.log({
                'mcc': mcc,
                'bin_miou': bin_iou.mean(),
                'cls_miou': all_ious.mean(),
            })

    return acc1_meter.avg, loss_meter.avg, miou_meter.avg, results


def main():
    # sweep setup
    log_to_wandb = not is_debug()

    # prepare args
    args, torch_gen = setup()
    early_stop_patience = 5
    steps = 5000
    out_folder = build_name(args)
    is_focal = False

    clip_grad = 0.5
    project_name = 'overleaf'

    im_size = args.im_size
    dataset = args.dataset
    device = args.device
    use_amp = args.amp
    use_fpn = args.use_fpn
    lr = args.lr
    label_smoothing = args.label_smoothing
    epochs = args.epochs
    scheduler_type = args.scheduler_type
    use_ass = args.use_ass
    only_polyp = args.only_polyp
    if only_polyp:
        use_ass = False

    model_config = args.config
    criterion_name = args.loss

    accum_iter = args.accum_iter
    batch_size = args.batch_size
    extra_losses = args.extra_losses
    num_workers = args.num_workers
    smooth_tvesky = 0.2
    decoder_name = args.decoder_name
    head_name = args.head_name

    loaders, dataset_classes = get_train_test_loaders(im_size, batch_size, dataset, num_workers, use_ass, only_polyp,
                                                      generator_seed=torch_gen)
    train_loader, val_loader, test_loader = loaders
    # add bg class to the corresponding number of classes
    dataset_classes += 1

    # log run to wandb if not debugging
    if log_to_wandb:
        wandb.init(project=project_name)

    model = build_models(im_size, use_fpn, decoder_name, head_name, model_config, dataset_classes, extra_losses,
                         verbose=True)
    model.to(device)

    # log run to wandb if not debugging
    if log_to_wandb:
        wandb.watch(model, log='all', log_freq=10)

    mixup_fn = None
    # Train
    criterion = Criterion(criterion_name, mixup_fn, label_smoothing, is_focal, smooth_tvesky, extra_losses, is_train=True, cls_weight=None,)
    val_criterion = Criterion(criterion_name, mixup_fn, label_smoothing, is_focal, smooth_tvesky=0., extra_losses=extra_losses, is_train=False, )
    criterion = criterion.to(device)

    print(f"Training model")
    print(steps, steps // len(train_loader), lr)

    optimizer = build_optimizer(model, lr)
    scaler = NativeScalerWithGradNormCount(clip_grad=clip_grad, scale=False)
    lr_scheduler = build_scheduler(optimizer, scheduler_type, steps, lr)
    early_stop = EarlyStopping(patience=early_stop_patience, save=True)

    out_folder = save_config(
        args,
        models=(model, None),
        datasets=(train_loader.dataset, test_loader.dataset),
        mixup_fn=mixup_fn,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=lr_scheduler,
        criterion=criterion
    )

    model = train(model, optimizer, scaler, lr_scheduler, criterion, train_loader, epochs, accum_iter,
                  val_loader=val_loader, mixup_fn=mixup_fn, use_amp=use_amp, num_classes=dataset_classes,
                  early_stop=early_stop, log_to_wandb=log_to_wandb, val_criterion=val_criterion)

    model = early_stop.load_best_weights(model)
    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
    }, "checkpoint.pth")
    save_model(model, f"{out_folder}")

    print("extract features")
    acc, loss_meter, val_miou, results = eval_model(test_loader, model, val_criterion, num_classes=dataset_classes,
                                                    wandb_log=log_to_wandb)

    for i in results:
        print(i)

    save_results(f"{out_folder}", results)
    del model,
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
