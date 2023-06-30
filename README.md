# PECAN

### Architecture

PECAN (Prediction Engine for the Cytotoxic Activity of Natural products) is a feedforward neural network that predicts cytotoxicity of compounds on 59 cell lines. It has an output of 356 (6 activity levels x 59 cell lines). A softmax over the 6 activity levels gives an activity level prediction for each of the 59 cell lines. Activity level predictions are given as numbers between 0 and 5.

| Number | Activity level   | -log GI50 values |
|--------|------------------|------------------|
|0       | Inactive         | &lt;4.1          |
|1       | Weakly active    | \[4.1, 5.0\)     |
|2       | Mildly active    | \[5.0, 6.0\)     |
|3       | Active           | \[6.0, 7.0\)     |
|4       | Potent           | \[7.0, 8.0\)     |
|5       | Super potent     | &geq;8.0         |


### Training

We trained on compounds from the NCI-60 dataset, which can be found [here](https://dtp.cancer.gov/dtpstandard/dwindex/index.jsp).

To train:

```
python3 train.py &lt;experiment name&gt; --data-path &lt;/path/to/data/&gt; --out &lt;/path/to/out/directory/&gt;
```

### Testing New Compounds

PECAN is fully trained and doesn't require extra training or fine tuning to be used to make predictions on new compounds.

To test compounds on PECAN, first convert the SMILE strings of any compounds to Morgan fingerprints using generate_fingerprints.py. This requires an environment with rdkit. To use the code as is, create a csv file with the ID or name of the compound followed by the SMILE string. Use a new line for each compound. 

Then run test.py using the Morgan fingerprints and loading the given checkpoint: weights.ckpy.

To test:

```
python3 test.py &lt;experiment name&gt; --load &lt;/path/to/checkpoint/weights.ckpt&gt;  --data-path &lt;/path/to/compounds/&gt; --out &lt;/path/to/out/directory/&gt;
```

### Citing PECAN

If you use PECAN in your work, please cite with the following BibTeX entry:

```bibtex
@software{PECAN2023,
    title        = {PECAN: Prediction Engine for the Cytotoxic Activity of Natural products},
    author       = {Martha Gahl and Hyunwoo Kim},
    year         = 2023,
    journal      = {GitHub repository},
    publisher    = {GitHub},
    howpublished = {\url{https://github.com/marthagahl/PECAN}}
}
```


