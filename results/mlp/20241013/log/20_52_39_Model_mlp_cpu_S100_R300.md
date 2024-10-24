```python
|2024-10-13 20:52:39| {
     'bs': 8, 'classification': False, 'dataset': cpu, 'debug': False,
     'decay': 0.0001, 'density': 0.8, 'device': cpu, 'device_name': core-i7-7820x,
     'epochs': 1000, 'experiment': False, 'inductive': False, 'log': <utils.logger.Logger object at 0x7fc1381052e0>,
     'logger': None, 'loss_func': L1Loss, 'lr': 0.001, 'model': mlp,
     'optim': AdamW, 'path': ./datasets/, 'patience': 100, 'program_test': False,
     'rank': 300, 'record': True, 'retrain': False, 'rounds': 5,
     'seed': 0, 'train_device': desktop-cpu-core-i7-7820x-fp32, 'train_size': 100, 'verbose': 0,
     'visualize': True,
}
|2024-10-13 20:52:39| ********************Experiment Start********************
|2024-10-13 20:52:40| MAE=0.0007 RMSE=0.0009 NMAE=0.1340 NRMSE=0.1652
|2024-10-13 20:52:41| MAE=0.0007 RMSE=0.0009 NMAE=0.1371 NRMSE=0.1690
|2024-10-13 20:52:41| MAE=0.0007 RMSE=0.0009 NMAE=0.1469 NRMSE=0.1818
|2024-10-13 20:52:41| MAE=0.0007 RMSE=0.0009 NMAE=0.1349 NRMSE=0.1659
|2024-10-13 20:52:41| MAE=0.0007 RMSE=0.0009 NMAE=0.1339 NRMSE=0.1737
|2024-10-13 20:52:41| ********************Experiment Results:********************
|2024-10-13 20:52:41| MAE: 0.0007 ± 0.0000
|2024-10-13 20:52:41| RMSE: 0.0009 ± 0.0000
|2024-10-13 20:52:41| NMAE: 0.1373 ± 0.0049
|2024-10-13 20:52:41| NRMSE: 0.1711 ± 0.0061
|2024-10-13 20:52:41| Acc_1: 0.0526 ± 0.0040
|2024-10-13 20:52:41| Acc_5: 0.2583 ± 0.0172
|2024-10-13 20:52:41| Acc_10: 0.4861 ± 0.0247
|2024-10-13 20:52:41| train_time: 1.2181 ± 0.4355
|2024-10-13 20:52:41| ********************Experiment Success********************
```

<div  align="center"> 
<img src="../fig/20_52_39_Model_mlp_cpu_S100_R300.pdf" 
width = "900" height = "800" 
alt="1" align=center />
</div>
