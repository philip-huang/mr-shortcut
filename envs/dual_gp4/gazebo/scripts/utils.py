import numpy as np
import matplotlib.pyplot as plt
import json
import time
from scipy.spatial.transform import Rotation as R

def load_json(fname):
	f = open(fname)
	content = json.load(f)
	f.close()
	return content

def write_json(graph, output_dir):
	json_graph = json.dumps(graph, indent=4)
	with open(output_dir, "w") as outfile:
		outfile.write(json_graph)
	outfile.close()
