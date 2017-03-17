#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse
from datetime import *

NCnt = 1000000
for i in range(NCnt) :
	x = random.random()
	y = 0.5 * x + (random.random() - 0.5) * 1
	if y > 0.5 * x :
		label = 1
	else :
		label = 0
	print i, label, "1:" + str(x), "2:" + str(y)
