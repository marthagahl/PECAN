import argparse
import torch
import os
import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from system import System

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='Name of the experiment')
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Number of samples per batch')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate per batch')
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--early-stop', type=int, default=100)
    parser.add_argument('--grad-acc', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--out', type=str, default='out', help='Path to save model checkpoint')
    args = parser.parse_args()

    model = System(args)

    early_stop_callback = EarlyStopping(monitor="validation loss", patience=args.early_stop)
    trainer = Trainer(
        callbacks=[early_stop_callback],
        enable_checkpointing=True,
        default_root_dir=os.path.join(args.out, args.name),
        accelerator = "cpu",
        max_epochs=args.max_epochs,
    )
    trainer.fit(model)
