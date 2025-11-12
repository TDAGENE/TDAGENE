# TDAGENE
We currently provide the main source code, and the complete code will be released soon.

## Dependencies

- Python == 3.8 
- Pytorch == 1.6.0
- scikit-learn==1.0.2
- numpy==1.20.3
- scanpy==1.7.2
- gseapy==0.10.8

- ## Usage

1. __Preparing  for gene expression profiles and  gene-gene adjacent matrix__
   
   TDAGENE integrates gene expression matrix __(N×M)__ with prior gene topology __(N×N)__ to learn low-dimensional vertorized representations with supervision.
2. **Command to run GENElink**
   
   To train an ab initio model, simply uses the script 'main.py'.
   
   `` python train.py``
   
   To apply MLP as score metric:
   
   Type == 'MLP', flag== False
   
