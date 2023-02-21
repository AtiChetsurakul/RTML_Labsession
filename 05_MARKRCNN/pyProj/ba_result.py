'''
root@0746646b7288:~/keep_lab/RTML_Labsession/05_MARKRCNN/pyProj# python3 train.py --use-cuda --iters 200 --dataset coco --data-dir /root/Datasets/coco
cuda:2
cuda: True
available GPU(s): 4
0: {'name': 'NVIDIA GeForce RTX 2080 Ti', 'capability': [7, 5], 'total_momory': 10.76, 'sm_count': 68}
1: {'name': 'NVIDIA GeForce RTX 2080 Ti', 'capability': [7, 5], 'total_momory': 10.76, 'sm_count': 68}
2: {'name': 'NVIDIA GeForce RTX 2080 Ti', 'capability': [7, 5], 'total_momory': 10.76, 'sm_count': 68}
3: {'name': 'NVIDIA GeForce RTX 2080 Ti', 'capability': [7, 5], 'total_momory': 10.76, 'sm_count': 68}

device: cuda:2
loading annotations into memory...
Done (t=22.77s)
creating index...
index created!
loading annotations into memory...
Done (t=0.63s)
creating index...
index created!
Namespace(ckpt_path='./maskrcnn_coco.pth', data_dir='/root/Datasets/coco', dataset='coco', device_num='2', epochs=3, iters=200, lr=0.00125, lr_steps=[6, 7], momentum=0.9, print_freq=100, results='./maskrcnn_results.pth', seed=3, use_cuda=True, warmup_iters=117266, weight_decay=0.0001)
Downloading: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth" to /root/.cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
100.0%

already trained: 0 epochs; to 3 epochs

epoch: 1
lr_epoch: 0.00125, factor: 1.00000
/usr/local/lib/python3.8/dist-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2157.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
0        0.737  0.569   0.038   0.012   0.290
100      0.693  0.177   0.171   0.085   0.312
iter: 96.4, total: 68.3, model: 35.7, backward: 20.2
iter: 159.5, total: 146.7, model: 131.4
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=4.73s).
Accumulating evaluation results...
DONE (t=0.68s).
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=0.99s).
Accumulating evaluation results...
DONE (t=0.66s).
accumulate: 7.1s
training: 19.3 s, evaluation: 43.1 s
{'bbox AP': 15.5, 'mask AP': 15.2}

epoch: 2
lr_epoch: 0.00125, factor: 1.00000
117300   0.243  0.077   0.067   0.031   0.217
117400   0.020  0.016   0.021   0.003   0.274
iter: 90.7, total: 75.2, model: 37.2, backward: 22.2
iter: 147.1, total: 134.2, model: 114.1
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.20s).
Accumulating evaluation results...
DONE (t=0.63s).
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=1.32s).
Accumulating evaluation results...
DONE (t=0.62s).
accumulate: 3.8s
training: 18.1 s, evaluation: 37.0 s
{'bbox AP': 19.7, 'mask AP': 16.6}

epoch: 3
lr_epoch: 0.00125, factor: 1.00000
234600   0.238  0.089   0.213   0.078   0.198
234700   0.228  0.090   0.219   0.028   0.227
iter: 98.3, total: 73.4, model: 35.3, backward: 23.2
iter: 141.9, total: 129.6, model: 114.2
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.99s).
Accumulating evaluation results...
DONE (t=0.71s).
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=1.17s).
Accumulating evaluation results...
DONE (t=0.63s).
accumulate: 3.5s
training: 19.7 s, evaluation: 35.8 s
{'bbox AP': 20.9, 'mask AP': 17.1}

total time of this training: 175.2 s
already trained: 3 epochs

root@0746646b7288:~/keep_lab/RTML_Labsession/05_MARKRCNN/pyProj# python3 train.py --use-cuda --iters 200000 --dataset coco --data-dir /root/Datasets/coco
cuda: True
available GPU(s): 4
0: {'name': 'NVIDIA GeForce RTX 2080 Ti', 'capability': [7, 5], 'total_momory': 10.76, 'sm_count': 68}
1: {'name': 'NVIDIA GeForce RTX 2080 Ti', 'capability': [7, 5], 'total_momory': 10.76, 'sm_count': 68}
2: {'name': 'NVIDIA GeForce RTX 2080 Ti', 'capability': [7, 5], 'total_momory': 10.76, 'sm_count': 68}
3: {'name': 'NVIDIA GeForce RTX 2080 Ti', 'capability': [7, 5], 'total_momory': 10.76, 'sm_count': 68}

device: cuda:2
loading annotations into memory...
Done (t=21.45s)
creating index...
index created!
loading annotations into memory...
Done (t=0.65s)
creating index...
index created!
Namespace(ckpt_path='./maskrcnn_coco.pth', data_dir='/root/Datasets/coco', dataset='coco', device_num='2', epochs=3, iters=200000, lr=0.00125, lr_steps=[6, 7], momentum=0.9, print_freq=100, results='./maskrcnn_results.pth', seed=3, use_cuda=True, warmup_iters=117266, weight_decay=0.0001)

already trained: 3 epochs; to 3 epochs

total time of this training: 0.0 s
root@0746646b7288:~/keep_lab/RTML_Labsession/05_MARKRCNN/pyProj# python3 train.py --use-cuda --iters 4000 --dataset coco --data-dir /root/Datasets/coco --epochs 5
cuda: True
available GPU(s): 4
0: {'name': 'NVIDIA GeForce RTX 2080 Ti', 'capability': [7, 5], 'total_momory': 10.76, 'sm_count': 68}
1: {'name': 'NVIDIA GeForce RTX 2080 Ti', 'capability': [7, 5], 'total_momory': 10.76, 'sm_count': 68}
2: {'name': 'NVIDIA GeForce RTX 2080 Ti', 'capability': [7, 5], 'total_momory': 10.76, 'sm_count': 68}
3: {'name': 'NVIDIA GeForce RTX 2080 Ti', 'capability': [7, 5], 'total_momory': 10.76, 'sm_count': 68}

device: cuda:2
loading annotations into memory...
Done (t=21.05s)
creating index...
index created!
loading annotations into memory...
Done (t=0.65s)
creating index...
index created!
Namespace(ckpt_path='./maskrcnn_coco.pth', data_dir='/root/Datasets/coco', dataset='coco', device_num='2', epochs=5, iters=4000, lr=0.00125, lr_steps=[6, 7], momentum=0.9, print_freq=100, results='./maskrcnn_results.pth', seed=3, use_cuda=True, warmup_iters=117266, weight_decay=0.0001)

already trained: 3 epochs; to 5 epochs

epoch: 4
lr_epoch: 0.00125, factor: 1.00000
/usr/local/lib/python3.8/dist-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2157.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
351800   0.269  0.008   0.136   0.038   0.190
351900   0.241  0.039   0.066   0.025   0.162
352000   0.144  0.074   0.030   0.040   0.276
352100   0.179  0.196   0.331   0.215   0.389
352200   0.241  0.094   0.426   0.241   0.271
352300   0.061  0.129   0.026   0.007   0.103
352400   0.169  0.078   0.214   0.032   0.199
352500   0.115  0.071   0.178   0.052   0.195
352600   0.092  0.177   0.338   0.169   0.265
352700   0.419  0.604   0.215   0.079   0.493
352800   0.127  0.041   0.137   0.040   0.299
352900   0.226  0.468   0.306   0.134   0.350
353000   0.031  0.052   0.019   0.004   0.096
353100   0.041  0.061   0.019   0.012   0.331
353200   0.121  0.118   0.317   0.212   0.401
353300   0.119  0.139   0.289   0.118   0.303
353400   0.080  0.368   0.056   0.012   0.150
353500   0.036  0.099   0.030   0.025   0.106
353600   0.218  0.344   0.486   0.235   0.391
353700   0.038  0.028   0.056   0.045   0.337
353800   0.053  0.039   0.153   0.111   0.367
353900   0.215  0.104   0.347   0.097   0.203
354000   0.681  2.025   0.795   0.105   0.524
354100   0.045  0.032   0.193   0.127   0.407
354200   0.093  0.061   0.314   0.058   0.297
354300   0.189  0.054   0.264   0.083   0.245
354400   0.134  0.061   0.108   0.031   0.436
354500   0.034  0.022   0.084   0.042   0.175
354600   0.061  0.007   0.016   0.006   0.191
354700   0.344  0.834   0.290   0.083   0.253
354800   0.178  0.416   0.059   0.022   0.134
354900   0.132  0.383   0.128   0.088   0.503
355000   0.150  0.092   0.335   0.258   0.420
355100   0.145  0.138   0.046   0.012   0.101
355200   0.258  0.135   0.261   0.108   0.208
355300   0.116  0.344   0.167   0.075   0.250
355400   0.156  0.248   0.095   0.056   0.330
355500   0.103  0.158   0.153   0.068   0.315
355600   0.135  0.031   0.137   0.051   0.148
355700   0.113  0.133   0.342   0.191   0.351
iter: 95.1, total: 75.6, model: 35.5, backward: 25.1
iter: 159.8, total: 139.4, model: 118.5
Loading and preparing results...
DONE (t=0.08s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=27.58s).
Accumulating evaluation results...
DONE (t=3.84s).
Loading and preparing results...
DONE (t=0.07s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=30.27s).
Accumulating evaluation results...
DONE (t=3.98s).
accumulate: 66.0s
training: 380.2 s, evaluation: 712.8 s
{'bbox AP': 20.0, 'mask AP': 16.0}

epoch: 5
lr_epoch: 0.00125, factor: 1.00000
469100   0.013  0.059   0.078   0.004   0.125
469200   0.025  0.038   0.031   0.013   0.118
469300   0.033  0.037   0.065   0.017   0.170
469400   0.014  0.009   0.020   0.002   0.349
469500   0.307  0.157   0.275   0.103   0.260
469600   0.050  0.069   0.075   0.018   0.288
469700   0.022  0.074   0.033   0.014   0.177
469800   0.219  0.348   0.200   0.086   0.332
469900   0.216  0.067   0.182   0.029   0.261
470000   0.161  0.334   0.301   0.115   0.362
470100   0.082  0.195   0.281   0.141   0.260
470200   0.129  0.061   0.221   0.191   0.370
470300   0.066  0.064   0.066   0.007   0.100
470400   0.030  0.031   0.005   0.003   0.101
470500   0.183  0.640   0.375   0.116   0.388
470600   0.024  0.018   0.054   0.044   0.279
470700   0.144  0.097   0.410   0.169   0.381
470800   0.237  0.707   0.282   0.148   0.304
470900   0.179  0.288   0.138   0.015   0.322
471000   0.015  0.022   0.044   0.015   0.147
471100   0.121  0.173   0.129   0.060   0.297
471200   0.071  0.078   0.158   0.091   0.336
471300   0.033  0.022   0.038   0.007   0.147
471400   0.036  0.056   0.010   0.011   0.446
471500   0.075  0.074   0.136   0.056   0.230
471600   0.048  0.121   0.033   0.015   0.227
471700   0.087  0.061   0.153   0.081   0.143
471800   0.216  0.019   0.082   0.007   0.143
471900   0.058  0.053   0.162   0.080   0.228
472000   0.033  0.013   0.039   0.015   0.229
472100   0.129  0.091   0.291   0.166   0.372
472200   0.280  0.485   0.568   0.168   0.330
472300   0.055  0.310   0.105   0.041   0.293
472400   0.135  0.052   0.080   0.042   0.302
472500   0.165  0.357   0.589   0.168   0.336
472600   0.225  0.040   0.146   0.030   0.353
472700   0.046  0.058   0.128   0.032   0.258
472800   0.148  0.028   0.182   0.068   0.261
472900   0.063  0.098   0.067   0.035   0.308
473000   0.128  0.414   0.049   0.033   0.284
iter: 97.5, total: 78.4, model: 36.8, backward: 26.3
iter: 151.1, total: 136.4, model: 115.3
Loading and preparing results...
DONE (t=0.10s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=30.14s).
Accumulating evaluation results...
DONE (t=4.19s).
Loading and preparing results...
DONE (t=0.09s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=31.85s).
Accumulating evaluation results...
DONE (t=4.24s).
accumulate: 70.7s
training: 389.9 s, evaluation: 681.0 s
{'bbox AP': 20.5, 'mask AP': 16.2}

total time of this training: 2165.4 s
already trained: 5 epochs
'''