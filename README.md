ARISE: Enhanced Ability and Privacy for Interdisciplinary Course Performance Prediction in Higher Education

This repository contains the PyTorch imploymentation for the ARISE.  If you have any questions on this repository or related paper, feel free to create an issue or send me an email.

Migrate code to the following environment:

Python 3.11

Torch 2.1.1

numpy 1.26.4

Scikit-learn 1.4.2

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

## Run 
The dataset configuration is located at lines 235-236 in the `client.py` file; determine the number of clients at line 222 based on the dataset you wish to run.

For the public dataset(tsu),  You can start the server in a terminal as follows:
```shell
python server.py
```
you will need to open five additional terminal windows and run the following commands.

Start clients 1 through 5 in the first to fifth terminals:

```shell
python client.py --partition-id 1
```
```shell
python client.py --partition-id 2
```
```shell
python client.py --partition-id 3
```
```shell
python client.py --partition-id 4
```
```shell
python client.py --partition-id 5
```
You will see that PyTorch is starting a federated training. 

For the private dataset (sau1,sau2),  You can start the server in a terminal as follows:
```shell
python server.py
```
you will need to open five additional terminal windows and run the following commands.

Start clients 1 and 2 in the two terminals:

```shell
python client.py --partition-id 1
```
```shell
python client.py --partition-id 2
```
You will see that PyTorch is starting a federated training. 

The non-personalized layer settings are located within two functions at lines 265 and 330 in the `client.py` file, which need to be configured accordingly, and also modify the model save path at line 246.
For example, the content at line 265 of the function should be set to.
```
params_to_upload = {
            'conv1.weight': net.conv1.weight.detach().cpu().numpy(),
            'conv1.bias': net.conv1.bias.detach().cpu().numpy(),
            'bn1.weight': net.bn1.weight.detach().cpu().numpy(),
            'bn1.bias': net.bn1.bias.detach().cpu().numpy(),
            'bn1.running_mean': net.bn1.running_mean.detach().cpu().numpy(),
            'bn1.running_var': net.bn1.running_var.detach().cpu().numpy(),
}
```
Consequently, the function at line 330 should be set to
```
parmname =['conv1.weight','conv1.bias','bn1.weight','bn1.bias','bn1.running_mean','bn1.running_var']
```
and the file save path should be
```
model_save_path = os.path.join('model', 'c1', f'client_{partition_id}_round_{config["round"]}.pth')
``` 

We will clean and release the code soon.  If you have any question, please feel free to contact me : chenxinyue@stu.sau.edu.cn
