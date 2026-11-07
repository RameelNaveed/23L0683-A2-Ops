"""Command line entry point for the training side.

    python -m trainer audit                        Task 1.2  dataset audit
    python -m trainer eda                          Task 1.3  figures
    python -m trainer split                        train / val / test lists
    python -m trainer preview                      Task 2.1  augmented batch figure
    python -m trainer train cnn_base               Tasks 2.2 to 2.4 (one preset from experiments.yaml)
    python -m trainer train --all                  every preset (Task 3.5)
    python -m trainer evaluate effb0_ft            Tasks 2.5 and 2.6 (test set, once)
    python -m trainer compare                      Task 3.5  run comparison
    python -m trainer tune toy | apply             Task 2.7
    python -m trainer gradcam effb0_ft             bonus
    python -m trainer register effb0_ft --alias champion     Task 3.6
    python -m trainer export                       copy the champion to models/champion.keras

Add --local to any MLflow command to keep the run on this machine.
"""
import argparse


def main():
    cli = argparse.ArgumentParser(prog="python -m trainer")
    sub = cli.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit")
    sub.add_parser("eda")
    sub.add_parser("split").add_argument("--seed", type=int, default=None)
    sub.add_parser("preview")

    tr = sub.add_parser("train")
    tr.add_argument("presets", nargs="*")
    tr.add_argument("--all", action="store_true")
    tr.add_argument("--max-epochs", type=int, default=None, help="cap every stage (quick runs)")
    tr.add_argument("--offline", action="store_true", help="random init instead of ImageNet (smoke tests only)")
    tr.add_argument("--local", action="store_true")

    ev = sub.add_parser("evaluate")
    ev.add_argument("preset")
    ev.add_argument("--fp-cost", type=float, default=3.0)
    ev.add_argument("--again", action="store_true", help="override the one look rule")
    ev.add_argument("--local", action="store_true")

    sub.add_parser("compare").add_argument("--local", action="store_true")

    tu = sub.add_parser("tune")
    tu.add_argument("part", choices=["toy", "apply"])
    tu.add_argument("--trials", type=int, default=8)
    tu.add_argument("--epochs", type=int, default=5)
    tu.add_argument("--offline", action="store_true")

    sub.add_parser("gradcam").add_argument("preset")

    rg = sub.add_parser("register")
    rg.add_argument("preset")
    rg.add_argument("--alias", default=None)
    rg.add_argument("--local", action="store_true")

    ex = sub.add_parser("export")
    ex.add_argument("--alias", default="champion")
    ex.add_argument("--local", action="store_true")

    args = cli.parse_args()

    if args.cmd == "audit":
        from trainer import audit
        audit.run()
    elif args.cmd == "eda":
        from trainer import eda
        eda.run()
    elif args.cmd == "split":
        from trainer import split
        split.run(*([args.seed] if args.seed is not None else []))
    elif args.cmd == "preview":
        from trainer import dataset
        dataset.preview()
    elif args.cmd == "train":
        from trainer import train
        from trainer.config import presets
        names = list(presets()) if args.all else args.presets
        if not names:
            cli.error("name a preset or pass --all")
        for n in names:
            train.run(n, args.max_epochs, args.local, "none" if args.offline else "imagenet")
    elif args.cmd == "evaluate":
        from trainer import evaluate
        evaluate.run(args.preset, args.fp_cost, args.again, args.local)
    elif args.cmd == "compare":
        from trainer import compare_runs
        compare_runs.run(args.local)
    elif args.cmd == "tune":
        from trainer import tune
        if args.part == "toy":
            tune.toy()
        else:
            tune.apply(args.trials, args.epochs, weights="none" if args.offline else "imagenet")
    elif args.cmd == "gradcam":
        from trainer import gradcam
        gradcam.run(args.preset)
    elif args.cmd == "register":
        from trainer import model_registry
        model_registry.register(args.preset, args.alias, args.local)
    elif args.cmd == "export":
        from trainer import model_registry
        model_registry.export(args.alias, args.local)


if __name__ == "__main__":
    main()
