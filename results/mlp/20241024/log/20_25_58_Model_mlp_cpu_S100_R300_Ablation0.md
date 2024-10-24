```python
|2024-10-24 20:25:58| {
     'Ablation': 0, 'bs': 8, 'classification': False, 'dataset': cpu,
     'debug': False, 'decay': 0.0001, 'density': 0.8, 'device': cpu,
     'device_name': core-i7-7820x, 'epochs': 1000, 'experiment': False, 'log': <utils.logger.Logger object at 0x7fdb800d2760>,
     'logger': None, 'loss_func': L1Loss, 'lr': 0.001, 'model': mlp,
     'optim': AdamW, 'path': ./datasets/, 'patience': 100, 'program_test': False,
     'rank': 300, 'record': True, 'retrain': True, 'rounds': 5,
     'seed': 0, 'train_device': desktop-cpu-core-i7-7820x-fp32, 'train_size': 100, 'verbose': 0,
}
|2024-10-24 20:25:58| ********************Experiment Start********************
|2024-10-24 20:26:06| Round=1 BestEpoch= 45 MAE=0.0007 RMSE=0.0009 NMAE=0.1340 NRMSE=0.1652 Training_time=0.9 s 
|2024-10-24 20:26:12| Round=2 BestEpoch= 40 MAE=0.0007 RMSE=0.0009 NMAE=0.1371 NRMSE=0.1690 Training_time=0.8 s 
|2024-10-24 20:26:19| Round=3 BestEpoch= 57 MAE=0.0007 RMSE=0.0009 NMAE=0.1469 NRMSE=0.1818 Training_time=1.1 s 
|2024-10-24 20:26:29| Round=4 BestEpoch=106 MAE=0.0007 RMSE=0.0009 NMAE=0.1349 NRMSE=0.1659 Training_time=2.0 s 
|2024-10-24 20:26:36| Round=5 BestEpoch= 58 MAE=0.0007 RMSE=0.0009 NMAE=0.1339 NRMSE=0.1737 Training_time=1.1 s 
|2024-10-24 20:26:36| ********************Experiment Results:********************
|2024-10-24 20:26:36| NMAE: 0.1373 ± 0.0049
|2024-10-24 20:26:36| NRMSE: 0.1711 ± 0.0061
|2024-10-24 20:26:36| MAE: 0.0007 ± 0.0000
|2024-10-24 20:26:36| RMSE: 0.0009 ± 0.0000
|2024-10-24 20:26:36| Acc_10: 0.4861 ± 0.0247
|2024-10-24 20:26:36| train_time: 1.1970 ± 0.4456
|2024-10-24 20:26:37| ********************Experiment Success********************
```

<div  align="center"> 
<img src="../fig/20_25_58_Model_mlp_cpu_S100_R300_Ablation0.pdf" 
width = "900" height = "800" 
alt="1" align=center />
</div>
