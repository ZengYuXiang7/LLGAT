```python
|2024-10-13 20:54:37| {
     'bs': 8, 'classification': False, 'dataset': cpu, 'debug': False,
     'decay': 0.0001, 'density': 0.8, 'device': cpu, 'device_name': core-i7-7820x,
     'epochs': 1000, 'experiment': False, 'inductive': False, 'log': <utils.logger.Logger object at 0x7f9be01452e0>,
     'logger': None, 'loss_func': L1Loss, 'lr': 0.001, 'model': mlp,
     'optim': AdamW, 'path': ./datasets/, 'patience': 100, 'program_test': False,
     'rank': 300, 'record': True, 'retrain': True, 'rounds': 5,
     'seed': 0, 'train_device': desktop-cpu-core-i7-7820x-fp32, 'train_size': 100, 'verbose': 0,
     'visualize': True,
}
|2024-10-13 20:54:37| ********************Experiment Start********************
|2024-10-13 20:54:45| Round=1 BestEpoch= 45 MAE=0.0007 RMSE=0.0009 NMAE=0.1340 NRMSE=0.1652 Training_time=1.0 s 
|2024-10-13 20:54:52| Round=2 BestEpoch= 40 MAE=0.0007 RMSE=0.0009 NMAE=0.1371 NRMSE=0.1690 Training_time=0.9 s 
|2024-10-13 20:55:00| Round=3 BestEpoch= 57 MAE=0.0007 RMSE=0.0009 NMAE=0.1469 NRMSE=0.1818 Training_time=1.2 s 
|2024-10-13 20:55:09| Round=4 BestEpoch=106 MAE=0.0007 RMSE=0.0009 NMAE=0.1349 NRMSE=0.1659 Training_time=2.0 s 
|2024-10-13 20:55:16| Round=5 BestEpoch= 58 MAE=0.0007 RMSE=0.0009 NMAE=0.1339 NRMSE=0.1737 Training_time=1.1 s 
|2024-10-13 20:55:16| ********************Experiment Results:********************
|2024-10-13 20:55:16| MAE: 0.0007 ± 0.0000
|2024-10-13 20:55:16| RMSE: 0.0009 ± 0.0000
|2024-10-13 20:55:16| NMAE: 0.1373 ± 0.0049
|2024-10-13 20:55:16| NRMSE: 0.1711 ± 0.0061
|2024-10-13 20:55:16| Acc_1: 0.0526 ± 0.0040
|2024-10-13 20:55:16| Acc_5: 0.2583 ± 0.0172
|2024-10-13 20:55:16| Acc_10: 0.4861 ± 0.0247
|2024-10-13 20:55:16| train_time: 1.2344 ± 0.4210
|2024-10-13 20:55:18| ********************Experiment Success********************
```

<div  align="center"> 
<img src="../fig/20_54_37_Model_mlp_cpu_S100_R300.pdf" 
width = "900" height = "800" 
alt="1" align=center />
</div>
