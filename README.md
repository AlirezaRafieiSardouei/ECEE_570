# ECEE_570
GitHub repository for the ECEE 570 course

## Train
To train a model from scratch, simply run:

```
python train.py --config CONFIG.yaml
```
where you replace `CONFIG.yaml` with the name of the configuration file you want to use.
We have included a configuration file under the folder `config/`.

For example, run:
```
python train.py --config ./config/aim_config.yaml
```
   

## Test
To test a model, you can also simply run:

```
python test.py --config CONFIG.yaml
```

where you need to specify the value of `TEST_EPOCH` in the `CONFIG.yaml` to evaluate a model trained after specific epochs, 
or you can also specify the value of `LOAD_PATH` to directly load a pre-trained checkpoint.
