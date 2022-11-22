# Inferior olive benchmark

## Usage

Running benchmarks (creating a benchmark.txt file).
Which Runners to benchmark is chosen at runtime.

```
$ git clone https://github.com/llandsmeer/iotf
$ cd iotf
$ python3 -m ioperf
```

Runnings tests (with optional graphcore tests specialization)

```
$ pytest ioperf --disable-warnings [-k graphcore]
[..]
2 passed, 11 deselected, 3 warnings in 6.03s
```

After making changes, run `bash build-docs.sh` and commit

## Module documentation

DOCS

## Authors

 - Max Engelen
 - Lennart Landsmeer
