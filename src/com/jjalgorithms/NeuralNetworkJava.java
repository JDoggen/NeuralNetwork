package com.jjalgorithms;


public class NeuralNetworkJava {
	
	public static void main(String[] args) {
		NeuralNetwork neuralNetwork = new NeuralNetwork(4, 1, 3, 4);
		double[] input = {0.1, 0.5, 0.9, 0.2};
		double[] output = {0.3, 0.8, 0.25, 0.5};
		
		for(int i = 0; i<1000; i++) {
			neuralNetwork.train(input, output);
		}
	}
}
