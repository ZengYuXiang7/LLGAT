```python
|2024-10-13 21:01:58| {
     'bs': 8, 'classification': False, 'dataset': cpu, 'debug': False,
     'decay': 0.0001, 'density': 0.8, 'device': cpu, 'device_name': core-i7-7820x,
     'epochs': 1000, 'experiment': False, 'inductive': False, 'log': <utils.logger.Logger object at 0x7f7b700c52e0>,
     'logger': None, 'loss_func': L1Loss, 'lr': 0.001, 'model': mlp,
     'optim': AdamW, 'path': ./datasets/, 'patience': 100, 'program_test': False,
     'rank': 300, 'record': True, 'retrain': True, 'rounds': 5,
     'seed': 0, 'train_device': desktop-cpu-core-i7-7820x-fp32, 'train_size': 100, 'verbose': 0,
     'visualize': True,
}
|2024-10-13 21:01:58| ********************Experiment Start********************
|2024-10-13 21:02:06| Round=1 BestEpoch= 45 MAE=0.0007 RMSE=0.0009 NMAE=0.1340 NRMSE=0.1652 Training_time=1.1 s 
|2024-10-13 21:02:13| Round=2 BestEpoch= 40 MAE=0.0007 RMSE=0.0009 NMAE=0.1371 NRMSE=0.1690 Training_time=0.9 s 
|2024-10-13 21:02:21| Round=3 BestEpoch= 57 MAE=0.0007 RMSE=0.0009 NMAE=0.1469 NRMSE=0.1818 Training_time=1.2 s 
|2024-10-13 21:02:30| Round=4 BestEpoch=106 MAE=0.0007 RMSE=0.0009 NMAE=0.1349 NRMSE=0.1659 Training_time=2.1 s 
|2024-10-13 21:02:38| Round=5 BestEpoch= 58 MAE=0.0007 RMSE=0.0009 NMAE=0.1339 NRMSE=0.1737 Training_time=1.1 s 
|2024-10-13 21:02:38| ********************Experiment Results:********************
|2024-10-13 21:02:38| MAE: 0.0007 ± 0.0000
|2024-10-13 21:02:38| RMSE: 0.0009 ± 0.0000
|2024-10-13 21:02:38| NMAE: 0.1373 ± 0.0049
|2024-10-13 21:02:38| NRMSE: 0.1711 ± 0.0061
|2024-10-13 21:02:38| Acc_1: 0.0526 ± 0.0040
|2024-10-13 21:02:38| Acc_5: 0.2583 ± 0.0172
|2024-10-13 21:02:38| Acc_10: 0.4861 ± 0.0247
|2024-10-13 21:02:38| train_time: 1.2899 ± 0.4029
|2024-10-13 21:02:39| ********************Experiment Success********************
```

<div  align="center"> 
<img src="../fig/21_01_58_Model_mlp_cpu_S100_R300.pdf" 
width = "900" height = "800" 
alt="1" align=center />
</div>
