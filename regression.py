import pandas as pd
import numpy as np
import matplotlib as plt

##Loading data


##Exploring the data
def explore(data):
	return 1


def cost_fn(b, m, data):
	#Err = mean sum over all points for distance
	error = 0
	len_data = len(data)

	#TODO: Vectorize!
	for i in range(0, len_data)
		x = data[i, 0]
		y = data[i, 1]
		error += (y - (m * x + b)) ** 2
	return error / float(len(data))


def step_gradient():
	
	b_gradient = 0
	m_gradient = 0
	n = len(data)
	N = float(n)

	#TODO: Vectorize!
	for i in range(0, n):
		x = data[i, 0]
		y = data[i, 1]
		b_gradient += -(2/N) * (y - ((m * x ) + b))
		m_gradient += -(2/N) * x * (y - ((m * x) + b))

	## "the gradient points in the direction of the greatest rate of increase of the function"
	## 	-- Wikipedia 
	## thats why there is a minus sign.

	n_b = b - (learning_rate * b_gradient)
	n_m = m - (learning_rate * m_gradient)

	return [n_b, n_m]


def run_gradient_descent(data, init_b, init_m, learning_rate, iterations):
	b = init_b
	m = init_m
	for i in range(num_iterations):
		b, m = step_gradient(b, m, array(data), learning_rate)
	return [b, m]


def run_linear_regression(data):
	#Hyper params:

	learning_rate = 0.0001;
	init_bias = 0;
	init_slope = 0;
	iterations = 1000;

	#Training model

	print('Started @ b = {0}, m = {1}'.format(init_bias, init_slope, cost_fn(init_bias, init_slope, data)))

	[b, m] = run_gradient_descent(data, init_bias, init_slope, learning_rate, iterations)

	print('Finished @ b = {0}, m = {1}'.format(init_bias, init_slope, cost_fn(init_bias, init_slope, data)))

def read_data():
	#Using genfromtxt instead of read_csv because
	#type ndarray
	training_set = np.genfromtxt("train-set.csv", delimiter=",", names="grades_1, grades_2, admitted")
	return training_set

if __name__ == '__main__':
	data = read_data()
	explore(data)
	run_linear_regression()