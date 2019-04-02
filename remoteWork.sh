#!/bin/bash

ssh -tt recovery1@10.8.0.1 ssh -N -f -L localhost:8887:localhost:8888 recovery1@192.168.1.106
