# Environment

You can use the following instructions to create the necessary environment for this framework.

```shell
conda install pytorch torchvision tqdm opencv
pip install tensorboardX
```

# Pytorch Docker

This docker is used for general training for Pytorch. You just need to pay attention to your network function, and avoid the trobules from writing underlying code.

If you want to begin a new project, you may need to rewrite Three part of things. And every part I have provided a "__demo"  file for you to copy.
## config

This section stores all the hyperparameters of the network you design.

We also provide a **root.json** file. If you need to deploy your code on multiple machines, but you suffer from the fact that the default data location and storage location of each machine are different, then this section can help you. 

The part marked with **%DATA%** or **%ROOT%** in the config file will automatically replaced with the location you recorded in **root.json**. 

Since your configuration file may be updated frequently, but the **root.json** file is not, this can save you a lot of trouble.

## dataset

This section records the loading ways of the datasets and the output methods.

## model

This section records the composition of the model, the calculation of the loss, and the evaluation index.

Note that your implementation of the loss and evaluate functions must inherit from the **weak_loss** and **weak_evaluate** virtual classes

#  Execute 

```shell
python train.py cfg_path ...
python test.py cfg_path ...
```

You can use -h to view command line parameters.

And any configuration parameter can be changed on the command line using the form below, although you cannot see it in -h.

```
-<cfg param name> <param>
```

## note

- if your -<cfg param name> does not exist in the **jsonc** file, it will not be appanded.

- if your <param> is a **list** / **dict**, use **' '** to mark the **str** type.
- if your <param> is a **dict**, it will only replace the item you mentioned, instead of overwriting the **original dict** with the **new dict** in the command line.

