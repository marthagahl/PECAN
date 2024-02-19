import argparse
import torch
import os
from pytorch_lightning import Trainer
from system import System

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('name', type=str, help='Name of the experiment')
        parser.add_argument('--load', type=str, required=True, help='Path to checkpoint weights')
        parser.add_argument('--data-path', type=str, required=True, help='Path to dataset')
        parser.add_argument('--fp-name', type=str, required=True, help='Name of file containing fingerprints')
        parser.add_argument('--batch-size', type=int, default=1000, help='Number of samples per batch')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate per batch')
        parser.add_argument('--max-epochs', type=int, default=1000)
        parser.add_argument('--max-steps', type=int, default=100000)
        parser.add_argument('--early-stop', type=int, default=50)
        parser.add_argument('--grad-acc', type=int, default=1)
        parser.add_argument('--num-workers', type=int, default=0)
        parser.add_argument('--val-interval', type=int, default=1.0)
        parser.add_argument('--out', type=str, default='out', help='Path to model predictions')
        args = parser.parse_args()


        model = System.load_from_checkpoint(args.load, args=args)
        trainer = Trainer()
        predictions = trainer.predict(model)

        cell_lines = open('cell_lines.txt','r')
        preds_out = open(args.out + 'PECAN_predictions.csv','w')
        preds_out.write(cell_lines.readline()+'\n')
        
        for b in range(len(predictions)):
            print (b)
            for p in predictions[b]:
                p = p.tolist()
                preds_out.write(str(p)[1:-1])
                preds_out.write('\n')

        preds_out.close()
        print ('Predictions written')

