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
python3 train.py <experiment name> --data-path </path/to/data/> --out </path/to/out/directory/>
```

### Testing New Compounds

PECAN is fully trained and doesn't require extra training or fine tuning to be used to make predictions on new compounds.

To test compounds on PECAN, first convert the SMILE strings of any compounds to Morgan fingerprints using generate_fingerprints.py. This requires an environment with rdkit. To use the code as is, create a csv file with the ID or name of the compound followed by the SMILE string. Use a new line for each compound. This will produce a pkl file. 

```
python3 generate_fingerprints.py </path/to/smile_strings.csv> </path/to/out/directory/> test.pkl
``` 

Then run test.py using the Morgan fingerprints and loading the given weights: weights.ckpt.

To test:

```
python3 test.py <experiment name> --load </path/to/checkpoint/weights.ckpt>  --data-path </path/to/compounds> --fp-name <name of fp file.pkl> --out </path/to/out/directory/>
```

### Citing PECAN

If you use PECAN in your work, please cite with the following BibTeX entry:

```bibtex
@software{PECAN2023,
    title        = {PECAN: Prediction Engine for the Cytotoxic Activity of Natural products},
    author       = {Martha Gahl and Hyunwoo Kim},
    year         = {2023},
    journal      = {GitHub repository},
    publisher    = {GitHub},
    howpublished = {https://github.com/marthagahl/PECAN}
}
```

The paper is now published and available [here](https://pubs.acs.org/doi/10.1021/acs.jnatprod.3c00879).

### Copyright Notice

This material is Copyright © 2023 The Regents of the University of California. All Rights Reserved. Permission to copy, modify, and distribute this material and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice, this paragraph and the following three paragraphs appear in all copies. Permission to make commercial use of this material may be obtained by contacting:


Office of Innovation and Commercialization  
9500 Gilman Drive, Mail Code 0910  
University of California  
La Jolla, CA 92093-0910  
innovation@ucsd.edu
 

This material and documentation are copyrighted by The Regents of the University of California. The material and documentation are supplied “as is”, without any accompanying services from The Regents. The end-user understands that the material was developed for research purposes.

 
IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS MATERIAL AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE MATERIAL PROVIDED HEREUNDER IS ON AN “AS IS” BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, IMPROVEMENTS, OR MODIFICATIONS.
