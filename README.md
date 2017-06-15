# fast-xnor-net
Hopefully fast implementation of XNOR-Net in C, because, why not?

## Dataset
http://yann.lecun.com/exdb/mnist/

## How to compile
```
mkdir build
cd build
cmake ..
cd ..
cmake --build build --config Release
```

## How to run
```
To run unoptimized normal Convolutional Neural Net:
./bin/FNC_XNORNET 0

To run unoptimized binary Convolutional Neural Net:
./bin/FNC_XNORNET 1

To run unoptimized XNOR Convolutional Neural Net:
./bin/FNC_XNORNET 2
```

## Project Structure
├── CMakeLists.txt

├── LICENSE

├── README.md

├── build

│   ├── CMakeCache.txt

│   ├── CMakeFiles

├── include

├── paper

│   └── 1603.05279.pdf

└── src

├── conv_layer.c

├── conv_layer.h

├── fully_con_layer.c

├── fully_con_layer.h

├── main.c

├── main.h

├── mnist_wrapper.c

├── mnist_wrapper.h

├── pool_layer.c

├── pool_layer.h

├── tensor.c

├── tensor.h

├── xnornet.c

└── xnornet.h
