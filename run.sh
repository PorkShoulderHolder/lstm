#!/bin/bash
wget http://cims.nyu.edu/~sfr265/model.net
luajit main.lua -mode generate -atom char -model model.net
