#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse
from datetime import *

NCnt = 100000
for i in range(NCnt) :
	x = random.random() * 2 - 1
	y = random.random() * 2 - 1

	if x * y >= 0 :
		label = 1
	else :
		label = 0

	print i, label, "1:" + str(x), "2:" + str(y)
