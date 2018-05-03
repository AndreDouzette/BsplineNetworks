import numpy as np

def housing(path = "./CaliforniaHousing/cal_housing.data", split = 0.3):
	data = np.loadtxt(path, delimiter = ",")
	rawInputs =  data[:, :-1]
	rawTargets = data[:, -1]
	samples = len(rawTargets)
	shuffling = np.arange(samples)
	np.random.shuffle(shuffling)
	inputs = (rawInputs[shuffling] - np.min(rawInputs, axis = 0))/(np.max(rawInputs, axis = 0) - np.min(rawInputs, axis = 0))
	targets = 0.5*(rawTargets[shuffling] - np.min(rawTargets))/(np.max(rawTargets) - np.min(rawTargets))
	targets = targets[:, np.newaxis]

	trainTargets = targets[int(split*samples):]
	testTargets = targets[:int(split*samples)]
	trainInputs = inputs[int(split*samples):]
	testInputs = inputs[:int(split*samples)]
	return trainInputs, trainTargets, testInputs, testTargets