# Method

Measured amount of loop bodies executed after 10 seconds.
Of course this was not exactly 10 seconds, so the
Measured time is displayed next to it.
No gap junctions.


```python
a = time.time()
ntimesteps = 0
while True:
    state = model(state)
    b = time.time()
    ntimesteps += 1
    if b - a > 10:
        break
print(ncells, ntimesteps, b - a)
```

Versions:

 - `poplar_sdk-ubuntu_18_04-2.5.1+1001-64add8f33d`
 - `tensorflow-2.5.2+gc2.5.1+193148+4673d3afb3b+intel_skylake512-cp36-cp36m-linux_x86_64.whl`

# Results

default: one IPU, fast math=False, keras model

identical cells
single timestep in python loop

```
ncells   iters seconds
======== ===== ==================
       1   361 10.017030477523804           # 1108x slower than realtime
      10   357 10.006501436233520
     100   339 10.014425516128540
    1000   301 10.028160095214844
   10000   280 10.011113643646240
  100000   131 10.029689550399780
 1000000    74 10.018659591674805
10000000    11 10.145269632339478
```

identical cells
40 timesteps (unrolled) in python loop

```
ncells   iters seconds            timesteps
======== ===== ================== =========
       1     8 10.172812223434448       320 # 1250x slower than realtime
      10     8 10.551693677902222       320
     100     8 10.365052461624146       320
    1000     7 10.353289127349854       280
   10000     6 10.475904226303100       240
  100000     3 10.380901098251343       120
 1000000     2 10.890616178512573        80
10000000     1 40.158017396926880        40
```

no keras model, straight python function
state = ipu.loops.repeat(40, timestep, state)
fastmath = True

```
ncells   iters seconds            timesteps
======== ===== ================== =========
       1    44 10.045462369918823      1760 # 227x slower than realtime
      10    44 10.046343088150024      1760
     100    39 10.078846216201782      1560 
    1000    37 10.124748706817627      1480
   10000    30 10.058508157730103      1200
  100000     6 10.004786252975464       240
 1000000     4 11.817041873931885       160
10000000     1 34.748606681823730        40
```


randomized g_CaL
single timestep in python loop

```
ncells   iters seconds
======== ===== ==================
       1   356 10.010644435882568
      10   352 10.023517608642578
     100   333 10.015699386596680
    1000   298 10.030692100524902
   10000   275 10.033795356750488
  100000   129 10.006984949111938
 1000000    74 10.000863313674927
10000000    11 10.461909532546997
```

16 IPU pairwise (select=16)
identical cells
single timestep in python loop

```
ncells   iters seconds
======== ===== ==================
       1   352 10.015149116516113
      10   354 10.024873256683350
     100   335 10.000465869903564
    1000   296 10.013682603836060
   10000   277 10.026544332504272
  100000   131 10.023094177246094
 1000000    75 10.028786420822144
10000000    11 10.489652633666992
```

one IPU
identical cells
fast math = true

```
ncells   iters seconds
======== ===== ==================
       1   352 10.001994371414185
      10   349 10.009512424468994
     100   330 10.014633417129517
    1000   296 10.019960165023804
   10000   274 10.006435632705688
  100000   130 10.018420934677124
 1000000    74 10.078833103179932
10000000    11 10.501261472702026
```
