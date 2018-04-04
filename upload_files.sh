#!/bin/bash

rsync -avzLPc --rsh='ssh -p9992' requirements.txt ball_on_plate_env.py neat plate.urdf $1 userquadro@uni1:~/neuroevol/
