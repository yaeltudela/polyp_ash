import os

import torch


def build_name(args, strict=True):

    dataset = args.dataset
    ass = "ASS" if args.use_ass else ""
    model_type = f"swin_{args.config}_{'FPN' if args.use_fpn else ''}"
    effective_b = args.batch_size * args.accum_iter

    out_folder = f"results/{dataset}_{ass}/{model_type}/eff_b{effective_b}{args.config}_{args.output}"
    base_folder = f"results/{dataset}_{ass}/{model_type}/eff_b{effective_b}{args.config}_{args.output}"

    if not strict:
        i = 1
        while os.path.exists(out_folder):
            out_folder = f"{base_folder}_{i}"
            i += 1

    return out_folder


def save_config(args, models=(), datasets=(), **kwargs):
    out_folder = build_name(args)
    os.makedirs(out_folder, exist_ok=True)
    to_save = [f"{k}: {v}\n" for k, v in args.__dict__.items()]
    with open(f"{out_folder}/config.txt", "w+") as f:
        for m in models:
            if m is None:
                continue
            print(m, file=f)
            f.writelines(['\n\n'])
        for d in datasets:
            if hasattr(d, 'transforms'):
                print(d, file=f)
                print(d.transforms, file=f)
            f.writelines(['\n\n'])

        f.writelines(to_save)
        f.writelines(['---------------------'])

        for k, v in kwargs.items():
            print(f"{k}: {v}\n", file=f)
        f.close()

    return out_folder


def save_model(model, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    torch.save(model.state_dict(), f"{out_folder}/model.pth")


def save_results(out_folder, results):
    os.makedirs(out_folder, exist_ok=True)
    with open(f"{out_folder}/res.txt", "w+") as f:
        for l in results:
            f.write(f"{l}\n")
        f.close()


