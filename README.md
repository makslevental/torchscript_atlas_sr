This repo demonstrates how to use [torchscript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) to wrap a pytorch model in a such that it can be embedded in a C++ program.

# Python

[trace_model.py](trace_model.py) loads a serialized pytorch model @ `model_fp` and performs source analysis on it in order to embed. It has a command-line interface, i.e., it can be run by executing 

`
$ python trace_model.py --model=dbpn --upscale=2
`

# CPP 

## Compiling
The CPP dependencies are [OpenCV 4.1.2](https://opencv.org/releases/) and [LibTorch](https://pytorch.org/cppdocs/installing.html) built against CUDA 10.1:

[![image](https://user-images.githubusercontent.com/5657668/69075562-9cc9c200-09ff-11ea-841f-c43be2e63f13.png)](https://pytorch.org/)

Adjust [CMakeLists.txt](CMakeLists.txt) appropriately according to your OpenCV and LibTorch install paths. Then

```shell script
$ cmake -DCMAKE_BUILD_TYPE=Debug
$ make
```

## Run
[main.cpp](main.cpp) is a CLI app that ingests a traced pytorch model and upscales [cat.jpg](cat.jpg). After compiling it can be run

```shell script
$ ./main \
  -w /home/maksim/dev_projects/torchscript/traced_dbpn_2x.pt \
  -i /home/maksim/dev_projects/torchscript/cat.jpg \
  -o /home/maksim/dev_projects/torchscript/sr_cat.jpg
```

where `-w` is the file path of the traced model as produced by [trace_model.py](#python).

