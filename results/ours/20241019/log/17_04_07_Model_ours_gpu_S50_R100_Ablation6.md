```python
|2024-10-19 17:04:07| {
     'Ablation': 6, 'bs': 4, 'classification': False, 'dataset': gpu,
     'debug': True, 'decay': 0.001, 'density': 0.8, 'device': cpu,
     'device_name': 1080Ti, 'epochs': 1, 'experiment': 0, 'graph_encoder': gat,
     'heads': 8, 'llm': 1, 'log': <utils.logger.Logger object at 0x7fb1b81b8310>, 'logger': None,
     'loss_func': L1Loss, 'lr': 0.001, 'model': ours, 'op_encoder': one_hot,
     'optim': AdamW, 'order': 3, 'path': ./datasets/, 'patience': 100,
     'program_test': False, 'rank': 100, 'record': 1, 'retrain': False,
     'rounds': 2, 'seed': 0, 'train_device': desktop-gpu-gtx-1080ti-fp32, 'train_size': 50,
     'verbose': 0,
}
|2024-10-19 17:04:07| ********************Experiment Start********************
|2024-10-19 17:04:15| Round=1 BestEpoch=  1 MAE=0.0016 RMSE=0.0018 NMAE=0.3171 NRMSE=0.3521 Training_time=0.1 s 
|2024-10-19 17:04:23| Round=2 BestEpoch=  1 MAE=0.0013 RMSE=0.0016 NMAE=0.2638 NRMSE=0.3108 Training_time=0.1 s 
|2024-10-19 17:04:23| ********************Experiment Results:********************
|2024-10-19 17:04:23| NMAE: 0.2905 ± 0.0266
|2024-10-19 17:04:23| NRMSE: 0.3315 ± 0.0206
|2024-10-19 17:04:23| MAE: 0.0014 ± 0.0001
|2024-10-19 17:04:23| RMSE: 0.0017 ± 0.0001
|2024-10-19 17:04:23| Acc_10: 0.1753 ± 0.0718
|2024-10-19 17:04:23| train_time: 0.0743 ± 0.0202
|2024-10-19 17:04:23| ********************Experiment Success********************
```

<div  align="center"> 
<img src="../fig/17_04_07_Model_ours_gpu_S50_R100_Ablation6.pdf" 
width = "900" height = "800" 
alt="1" align=center />
</div>
