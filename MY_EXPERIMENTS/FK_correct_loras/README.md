## Feynman-Kac Corrector Experiment

Simply edit the ```config.py``` and run ```sbatch job.sh```.  The model loaded (SDXL 1.0 base model) shall denoise using by fusing the scores predicted by the provided LoRAs, while also keeping the ```log_weights``` of the generation process. 

The results will be in the ```output``` folder, in which ```images``` stores the generated images, and ```results.csv``` stores the prompts used, hyperparameters $\beta , K$, as well as the log-weights of each image.
