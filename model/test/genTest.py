#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse
from datetime import *

D = 20
J = 5
N = 100000
Q = 0.1

for i in range(N) :
	x = [random.random() for i in range(D)]
	t = sum(x[:J])
	if t > J / 2.0 :
		p = 1 - Q
	else :
		p = Q

	if random.random() > p :
		y = 1
	else : 
		y = 0

	print i, y, "\t".join([str(i) + ":" + str(x[i]) for i in range(len(x)) if x[i] != 0])
