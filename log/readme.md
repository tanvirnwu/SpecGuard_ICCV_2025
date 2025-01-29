# Command

We trained and tested WaveGuard on a CelebHQ [1] subset adding different noises and found that the model converged on 208 number epochs. The validation data is also provided in the "log" folder in CSV format and the decoder loss curve.

If you are still interested in testing, please replace the dataset path and run with the following code: 

```
python main.py new --data-dir [CelebHQ] --batch-size 32 --epochs 300 --name "celebhq-subset" --size 256 --message 128 --watermark-radius 50 --tensorboard --enable-fp16 --noise 'crop((0.2,0.3),(0.4,0.5))+cropout((0.11,0.22),(0.33,0.44))+dropout(0.2,0.3)+jpeg()'
````

Upon acceptance, we will further enhance the documentation to provide exact information to run WaveGuard to achieve the best performance.


[1] Karras, Tero. "Progressive Growing of GANs for Improved Quality, Stability, and Variation." arXiv preprint arXiv:1710.10196 (2017).
