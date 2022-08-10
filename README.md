# Quick-CapsNet (QCN)
This is the code for Quick-CapsNet paper accepted in AICCSA-2020.

The base code for this project is the following: <br />
https://github.com/gram-ai/capsule-networks


# Installation
Step 1. Install PyTorch and Torchvision:<br />
`conda install -c pytorch pytorch torchvision`<br />
Step 2. Install Torchnet: <br />
`pip install torchnet`

# Usage
The "Main.py" file trains the network and prints the results to the files in the specified folder (input args). <br />
Parameters:<br />
`--dset`: Choice of dataset (options: MNIST, F-MNIST, SVHN and CIFAR-10)<br />
`--nc`: Number of classes in the chosen dataset<br />
`--w` : The width/height of input images<br />
`--bsize`: Batch size<br />
`--ne`: Number of epochs to train the model<br />
`--lr`: Initial Learning rate for the Adam optimizer <br />
`--niter`: Number of iterations for DR algorithm<br />
`--npc`: Number of PCs generated by the Fully-Connected layer <br />

`--ich`: number of channels in the input image<br />
`--dec_type`: The type of decoder used (options: FC, DECONV)<br />
`--res_folder`: The output folder to print the results into<br />
`--nc_recon`: Performing the reconstruction in a single channel or all channels (options: 1,3)<br />
`--hard`: Perform hard-training at the end or not (hard-training: training while tightening the bounds of the margin loss, options: 0,1)<br />
`--checkpoint`: The file address of the checkpoint file (used for hard training) <br />
`--test_only`: Only loads the checkpoint and tests the network on the test set (options: 0, 1)

