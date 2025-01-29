# Command

We against trained and tested WaveGuard on a CelebHQ [1] subset adding different noises and found that the model converged on 208 number epochs. The validation 

```
python main.py new --data-dir [CelebHQ] --batch-size 32 --epochs 300 --name "celebhq-subset" --size 256 --message 128 --watermark-radius 50 --tensorboard --enable-fp16 --noise 'crop((0.2,0.3),(0.4,0.5))+cropout((0.11,0.22),(0.33,0.44))+dropout(0.2,0.3)+jpeg()'
````


[1] 
