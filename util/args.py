import argparse
import os

from torch.utils.tensorboard import SummaryWriter


def init_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", type=str, required=False)
    parser.add_argument("--input-delimiter", type=str, default=",", required=False)
    parser.add_argument("--validation-path", type=str, required=False)
    parser.add_argument("--test-path", type=str, required=False)
    parser.add_argument("--model-path", type=str, required=False)
    parser.add_argument("--output-dir", type=str, required=False)
    parser.add_argument("--target-items-path", type=str, required=False, nargs="?", const="")
    parser.add_argument("--evaluation-only", action="store_true", default=False, help="evaluation only. No training")
    parser.add_argument(
        "--do-evaluation", action="store_true", default=False, help="if true, evaluate after embeddings are saved"
    )
    parser.add_argument(
        "--save-user-doc-preds",
        action="store_true",
        default=False,
        help="if true, save user doc predictions (required for visualization)",
    )
    parser.add_argument(
        "--load-format", type=str, default="npy", choices=["npy", "binary", "text"], help="format to load embeddings"
    )
    parser.add_argument("--save-npy", action="store_true", default=False, help="save embeddings in npy")
    parser.add_argument("--save-text", action="store_true", default=False, help="save embedding in text")
    parser.add_argument("--save-binary", action="store_true", default=False, help="save embedding in binary")
    parser.add_argument(
        "--save-sequential",
        action="store_true",
        default=False,
        help="if true, dont save text/binary embeddings in parallel (useful for memory purpose)",
    )
    parser.add_argument(
        "--save-io-workers",
        type=int,
        default=8,
        help="number of workers for saving text/binary embeddings in parallel",
    )
    parser.add_argument("--logging-steps", type=int, default=10, required=False)
    parser.add_argument("--splatt-config", type=str, help="destination to store splatt config for training")

    parser.add_argument("--factors", type=int, default=96)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--user-top-k", type=int, default=50)
    parser.add_argument("--test-knn", action="store_true", default=False, help="test with exact KNN search")
    parser.add_argument("--distance-func", type=str, default="ip", choices=["ip", "cosine"])
    parser.add_argument(
        "--export-normalized-embedding",
        action="store_true",
        help="if true, will l2 normalize user and item embedidngs before saving them",
    )
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--num-save-steps", type=float, default=0.5)
    parser.add_argument(
        "--ckpts_size_limit",
        type=int,
        default=5,
        help="max number of checkpoints saved, for saving disk, -1 means save all checkpoints",
    )
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--dlis-iterations", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=None, help="gain percent threshold for convergence test")
    parser.add_argument(
        "--epsilon-ts", type=int, default=None, help="timespan threshold for gain percent in convergence test"
    )
    parser.add_argument("--max-time", type=int, default=None, help="max allowed training time before termination")
    parser.add_argument(
        "--conv-metric", type=str, default="faiss_knn_recall_origin_50", help="the metric for convergence test"
    )
    parser.add_argument("--save-per-epochs", type=int, default=50)

    parser.add_argument(
        "--ckpt-init-dir", type=str, required=False, nargs="?", const="", help="checkpoint folder path on cosmos"
    )
    parser.add_argument("--rating-scale", type=float, default=1.0, required=False)
    parser.add_argument("--rating-offset", type=float, default=0.0, required=False)
    parser.add_argument(
        "--weighting-method",
        type=str,
        default="binary_weight",
        required=False,
        help="weighting method for train data. Possible options are binary_weight or frequency_weight or ips_weight or bm25_weight",
    )
    parser.add_argument(
        "--read-frequency",
        action="store_true",
        default=False,
        help="if true, use frequency of clicks (otherwise will not consider frequencies). Frequencies are used in bm25_weight and frequency_weight and log_frequency_weight, but are ignored in ips_weight and binary_weight.",
    )
    parser.add_argument("--ips-power", type=float, default=0.5, required=False, help="used in ips_weight")
    parser.add_argument("--K1", type=float, default=100, required=False, help="used in bm25_weight_local")
    parser.add_argument("--B", type=float, default=0.5, required=False, help="used in bm25_weight_local")
    parser.add_argument(
        "--load-training-method",
        type=str,
        default="load_train_data",
        required=False,
        help="if true, will time decay the user-item interaction matrix. Possible options are load_train_data_weighted_by_time and load_train_data",
    )
    parser.add_argument(
        "--range-max", type=float, default=2, required=False, help="used in load_train_data_weighted_by_time"
    )
    parser.add_argument(
        "--range-min", type=float, default=1, required=False, help="used in load_train_data_weighted_by_time"
    )
    parser.add_argument(
        "--power", type=float, default=1.0, required=False, help="used in load_train_data_weighted_by_time"
    )

    # parameter to control the returned model
    parser.add_argument(
        "--return-model",
        type=str,
        default="final",
        choices=["best", "final"],
        help="the model returned after training",
    )

    parser.add_argument(
        "--parallel_valid",
        action="store_true",
        default=False,
        help="whether to make validation parallel with training (this will double memory usage)",
    )

    parser.add_argument(
        "--reco-cutoff-score",
        type=float,
        default=None,
        help="threshold for cutting off recommendations. If is not None, will throw away recommendation which have distance greater/equal than this threshold",
    )

    parser.add_argument(
        "--expand-user-coverage", action="store_true", help="if true, expand user coverage to non-prism users"
    )
    parser.add_argument(
        "--input-path-expandusercoverage",
        type=str,
        default="",
        help="if expand_user_coverage is True, use this file as training set",
    )
    parser.add_argument(
        "--test-path-uniqueextra",
        type=str,
        default="",
        help="if expand_user_coverage is True, use this file to save user_doc_preds_for_extrausers.tsv for non-prism users",
    )
    parser.add_argument(
        "--test-path-uniquespecific",
        type=str,
        default="",
        help="if expand_user_coverage is True, use this file to save user_doc_preds_for_specificusers.tsv for specific users",
    )

    parser.add_argument(
        "--doc-embeddings-visual-path", type=str, default="", help="file path of visual doc embeddings"
    )
    parser.add_argument("--doc-embeddings-text-path", type=str, default="", help="file path of text doc embeddings")
    parser.add_argument("--report-diversity", action="store_true", default=False, help="report diversity metric")
    parser.add_argument("--report-relevancy", action="store_true", default=False, help="report relevancy metric")
    parser.add_argument("--report-soft-recall", action="store_true", default=False, help="report soft-recall metric")

    parser.add_argument("--save-log-to-file", action="store_true", default=False, help="save log to file")
    parser.add_argument("--tensorboard", action="store_true", help="tensorboard")
    args, _ = parser.parse_known_args('testin g')

    assert (
        args.epoch % args.logging_steps == 0 or args.epoch < args.logging_steps
    ), "args.epoch should be divisible by args.logging_steps or greater than logging steps to bypass evaluation in training"

    if args.tensorboard:
        args.tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
        args.writer = SummaryWriter(log_dir=args.tensorboard_dir, comment="comment")
    else:
        args.writer = None

    if not args.model_path:
        args.model_path = args.output_dir

    return args
