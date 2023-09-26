import atexit
import os
import sys
import logging
import argparse
import time
from collections import defaultdict

from matplotlib import pyplot as plt
import torch.optim
import torch_geometric as pyg
from pytorch3d.loss import chamfer_distance
from torch import nn

from highlevel_planning_py.predicate_learning import training_utils as tu
from highlevel_planning_py.predicate_learning import dataset_pointcloud
from highlevel_planning_py.predicate_learning import models_pc_encoding as mdl
from highlevel_planning_py.tools_pl import util

SRCROOT = os.path.join(os.path.dirname(__file__), "..")
PATHS = {
    "": "",
    "data_dir": os.path.join(os.path.expanduser("~"), "Data", "highlevel_planning"),
    "asset_dir": os.path.join(SRCROOT, "data", "models"),
    "igibson_dir": os.path.join(os.path.expanduser("~"), "Data", "igibson", "assets"),
}


def parser_arguments():
    parser = argparse.ArgumentParser()
    tu.parse_general_args(parser)
    parser.add_argument("--num_epochs", action="store", type=int, required=True)
    parser.add_argument(
        "--model_size_enc",
        action="store",
        type=str,
        choices=["original", "large", "small"],
        help="This specifies the model size",
        default="original",
    )
    parser.add_argument(
        "--model_size_dec",
        action="store",
        type=str,
        choices=["original", "large", "small"],
        help="This specifies the model size",
        default="original",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        action="store",
        choices=["pointnet2", "conv"],
        required=True,
    )
    parser.add_argument("--encoding_dim", type=int, default=128)  # Same as Achlioptas
    parser.add_argument(
        "--decoder_type",
        type=str,
        action="store",
        choices=["pointnet2", "conv", "fc"],
        required=True,
    )
    parser.add_argument(
        "--dataset", type=str, choices=["own", "shapenet"], required=True
    )
    parser.add_argument(
        "--dataset_normalization",
        type=str,
        choices=[
            "none",
            "individual_max",
            "all_data_median",
            "all_data_mean",
            "all_data_max",
        ],
        required=True,
    )
    parser.add_argument("--data_prescale_target", type=float, default=1.0)
    parser.add_argument(
        "--augment_training_data", type=util.string_to_list, default=None
    )
    parser.add_argument("--save_model_interval", type=int, default=100)
    parser.add_argument("--save_vis_interval", type=int, default=20)
    parser.add_argument("--test_dataset_size", type=int, default=-1)
    parser.add_argument("--evaluate_every", type=int, default=10)
    parser.add_argument("--initialize_with", type=str, default="")
    args = parser.parse_args()
    return args


def get_dataset_config(
    dataset: str,
    train: bool,
    args,
    device,
    predicate_name,
    dataset_dir,
    dataset_size,
    augment_training_data=None,
    test_dataset_label=None,
    given_scale=None,
    scale_target: float = 1.0,
    dataset_normalization="none",
):
    if augment_training_data is not None:
        augment_training_data = tuple(augment_training_data)
        assert len(augment_training_data) == 2
        assert augment_training_data[0] <= augment_training_data[1]
    if dataset == "own":
        custom = {
            "num_points": 2048,
            "given_scale": given_scale,
            "augment_scale": augment_training_data,
            "scale_target": scale_target,
        }
        dataset_type = dataset_pointcloud.SinglePointcloudDataset
    elif dataset == "shapenet":
        custom = {
            "shapenet_dir": os.path.join(
                os.path.expanduser("~"),
                "Data",
                "latent_3d_points",
                "data",
                "shape_net_core_uniform_samples_2048_split",
            ),
            "train": train,
            "augment_scale": augment_training_data,
        }
        dataset_type = dataset_pointcloud.SinglePointcloudDatasetShapenet
    else:
        raise ValueError
    if train:
        splits = {"train": 0.9, "val": 0.1}
    else:
        splits = {test_dataset_label: 1.0}
    dataset_config = tu.DatasetConfig(
        device,
        dataset_dir,
        args.dataset_id,
        predicate_name,
        splits=splits,
        target_model="enc",
        use_tensors=True,
        dataset_size=dataset_size,
        normalization_method=dataset_normalization,
        custom=custom,
    )
    return dataset_config, dataset_type


def single_epoch(
    epoch: int,
    final_epoch: int,
    label: str,
    loader,
    encoder: nn.Module,
    decoder: nn.Module,
    logger: tu.TrainLogger,
    device: str,
    optimizer,
    out_dir,
    args,
    force_test=False,
):
    if "val" in label or "test" in label or force_test:
        train = False
        encoder.eval()
        decoder.eval()
    elif label == "train":
        train = True
        assert optimizer is not None
        encoder.train()
        decoder.train()
    else:
        raise ValueError

    cloud_in, cloud_out, bs, num_points = None, None, None, None
    for batch in loader:
        cloud_in = batch.to(device)
        encoding = encoder.forward(cloud_in)
        cloud_out = decoder.forward(encoding)

        bs = batch.num_graphs
        num_points = int(cloud_out.shape[0] / bs)

        loss_value = (
            0.5
            * chamfer_distance(
                cloud_in.pos.view(bs, num_points, 3),
                cloud_out.view(bs, num_points, 3),
                point_reduction="sum",
            )[0]
        )

        if train:
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        logger.record_iteration(
            loss={"chamfer": loss_value.item()},
            lr=optimizer.defaults["lr"],
            batch_size=bs,
        )
    logger.write_epoch(epoch)

    if train and tu.is_nth_epoch(epoch, args.save_model_interval, final_epoch):
        # Save models
        filename = os.path.join(out_dir, "models", f"encoder_{epoch:03}.pt")
        torch.save(encoder.state_dict(), filename)
        filename = os.path.join(out_dir, "models", f"decoder_{epoch:03}.pt")
        torch.save(decoder.state_dict(), filename)

    if tu.is_nth_epoch(epoch, args.save_vis_interval, final_epoch):
        # Save sample cloud
        pc_in = cloud_in.pos.view(bs, num_points, 3)[0].detach().cpu().numpy()
        pc_out = cloud_out.view(bs, num_points, 3)[0].detach().cpu().numpy()

        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.scatter(pc_in[:, 0], pc_in[:, 1], pc_in[:, 2])
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.scatter(pc_out[:, 0], pc_out[:, 1], pc_out[:, 2])

        plt.savefig(os.path.join(out_dir, f"samples_{label}", f"sample_{epoch:03}"))
        plt.close(fig)


def main(args: argparse.Namespace):
    # Config
    if args.gpu:
        device = "cuda"
        assert torch.cuda.is_available()
    else:
        device = "cpu"
    predicate_dir = os.path.join(PATHS["data_dir"], "predicates")
    dataset_dir = os.path.join(predicate_dir, "data")

    resume_training = args.resume
    resume_timestring = args.id_string

    out_dir = tu.setup_out_dir(
        predicate_dir,
        resume_training,
        resume_timestring,
        screen_output=args.screen_output,
        training_type="pcenc",
    )
    logging.info(f"Command: {' '.join(sys.argv)}")
    pyg.seed_everything(args.random_seed)

    if not resume_training:
        tu.save_parameters(
            vars(args), "arguments_training", out_dir, parameters_obj=args
        )
    else:
        tu.save_parameters(
            vars(args), "arguments_training_res", out_dir, parameters_obj=args
        )

    if not resume_training:
        stats = {"time_elapsed": 0.0, "total_iterations": 0, "num_sessions": 0}
    else:
        stats = tu.load_parameters(out_dir, "stats")

    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)

    # Training dataset config
    dataset_configs, dataset_types = list(), list()
    dataset_config, dataset_type = get_dataset_config(
        args.dataset,
        train=True,
        args=args,
        device=device,
        predicate_name=args.predicate_name,
        dataset_dir=dataset_dir,
        dataset_size=args.dataset_size,
        augment_training_data=args.augment_training_data,
        scale_target=args.data_prescale_target,
        dataset_normalization=args.dataset_normalization,
    )
    dataset_configs.append(dataset_config)
    dataset_types.append(dataset_type)

    # Create training dataset
    loaders = defaultdict(dict)
    datasets = defaultdict(dict)
    dimensions = dict()
    tu.setup_data(
        loaders,
        datasets,
        dimensions,
        resume_training,
        out_dir,
        args.batch_size,
        dataset_configs,
        dataset_types,
    )

    # Test dataset config
    dataset_test_configs, dataset_test_types = list(), list()
    dataset_config, dataset_type = get_dataset_config(
        "shapenet",
        train=False,
        args=args,
        device=device,
        predicate_name=args.predicate_name + "_test",
        dataset_dir=dataset_dir,
        dataset_size=args.test_dataset_size,
        test_dataset_label="test_shapenet",
    )
    dataset_test_configs.append(dataset_config)
    dataset_test_types.append(dataset_type)

    dataset_config, dataset_type = get_dataset_config(
        "own",
        train=False,
        args=args,
        device=device,
        predicate_name=args.predicate_name + "_test",
        dataset_dir=dataset_dir,
        dataset_size=args.test_dataset_size,
        test_dataset_label="test_own",
        scale_target=0.5,
        dataset_normalization="all_data_max",
    )
    dataset_test_configs.append(dataset_config)
    dataset_test_types.append(dataset_type)

    # Create test datasets
    tu.setup_data(
        loaders,
        datasets,
        dimensions,
        resume_training,
        out_dir,
        args.batch_size,
        dataset_test_configs,
        dataset_test_types,
    )

    for label, loader in loaders["enc"].items():
        stats[f"{label}_dataset_size"] = loader.sampler.num_samples
        os.makedirs(os.path.join(out_dir, f"samples_{label}"), exist_ok=True)

    # Model
    if args.encoder_type == "pointnet2":
        if args.decoder_type in ["pointnet2"]:
            encoder = mdl.Pointnet2Encoder(
                args.encoding_dim, args.model_size_enc, autoencoder=True
            )
        else:
            encoder = mdl.Pointnet2Encoder(
                args.encoding_dim, args.model_size_enc, autoencoder=False
            )
    elif args.encoder_type == "conv":
        encoder = mdl.ConvEncoder(args.encoding_dim, model_config=args.model_size_enc)
    else:
        raise ValueError
    if args.decoder_type == "pointnet2":
        assert (
            args.encoder_type == "pointnet2"
        ), "Pointnet decoder only works with pointnet encoder."
        decoder = mdl.Pointnet2Decoder(args.encoding_dim, args.model_size_dec)
    elif args.decoder_type == "conv":
        decoder = mdl.ConvDecoder(
            args.encoding_dim,
            datasets["enc"]["train"].num_points,
            model_config=args.model_size_dec,
        )
    elif args.decoder_type == "fc":
        decoder = mdl.FullyConnectedDecoder(
            args.encoding_dim,
            datasets["enc"]["train"].num_points,
            model_config=args.model_size_dec,
        )
    else:
        raise ValueError
    stats["encoder_num_parameters"] = pyg.graphgym.utils.comp_budget.params_count(
        encoder
    )
    stats["decoder_num_parameters"] = pyg.graphgym.utils.comp_budget.params_count(
        decoder
    )

    # Initialize with trained weights, in case of fine-tuning
    if len(args.initialize_with) > 0 or resume_training:
        if len(args.initialize_with) > 0:
            model_dir = os.path.join(
                predicate_dir, "training", args.initialize_with, "models"
            )
        else:
            model_dir = os.path.join(out_dir, "models")
        model_list = os.listdir(model_dir)
        model_list_enc = sorted([f for f in model_list if "enc" in f])
        model_list_dec = sorted([f for f in model_list if "dec" in f])
        state_file_enc = os.path.join(model_dir, model_list_enc[-1])
        state_file_dec = os.path.join(model_dir, model_list_dec[-1])
        encoder.load_state_dict(torch.load(state_file_enc))
        decoder.load_state_dict(torch.load(state_file_dec))

    encoder.to(device)
    decoder.to(device)

    # Store parameters relevant for inference
    tu.save_parameters(
        {
            "data_scaling": datasets["enc"]["train"].scale,
            "num_points": datasets["enc"]["train"].num_points,
        },
        "meta_parameters",
        os.path.join(out_dir, "models"),
    )

    # Training utils
    params_encoder = list(filter(lambda p: p.requires_grad, encoder.parameters()))
    params_decoder = list(filter(lambda p: p.requires_grad, decoder.parameters()))
    optim = torch.optim.Adam(params_encoder + params_decoder, lr=args.learning_rate)
    loggers = {
        label: tu.TrainLogger(label, out_dir, loss_aggregation="mean")
        for label in loaders["enc"]
    }

    logging.info("Start training")
    start_time = time.time()
    atexit.register(tu.exit_handler, start_time, stats, out_dir, loggers)

    # Evaluate once before first epoch if initializing network
    starting_epoch = 0 if len(args.initialize_with) > 0 else 1

    final_epoch = args.num_epochs
    if resume_training:
        starting_epoch += stats["total_iterations"]
        final_epoch += stats["total_iterations"]

    for epoch in range(starting_epoch, final_epoch + 1):
        force_test = True if epoch == 0 else False
        for label in loaders["enc"]:
            if (label != "train") and not tu.is_nth_epoch(
                epoch, args.evaluate_every, final_epoch
            ):
                continue
            single_epoch(
                epoch,
                final_epoch,
                label,
                loaders["enc"][label],
                encoder,
                decoder,
                loggers[label],
                device,
                optim,
                out_dir,
                args,
                force_test=force_test,
            )
        stats["total_iterations"] += 1
        if epoch > 0:
            avg_time_per_epoch = (time.time() - start_time) / epoch
            logging.info(f"Avg time per epoch: {avg_time_per_epoch}")

    end_time = time.time()
    stats["time_elapsed"] += end_time - start_time
    stats["num_sessions"] += 1
    tu.save_parameters(stats, "stats", out_dir)
    logging.info("Saved stats")


if __name__ == "__main__":
    arguments = parser_arguments()
    main(arguments)
