
import java.io.File;
import java.util.Scanner;
import java.io.FileNotFoundException;

public class ThreeLayer 
{	

	private static int inputPixels = 784;

	public static void main(String[] args) throws FileNotFoundException
	{  
		
		int batchSize = 100;
		int neurons1 = 40;
		int outputs = 10;
		double k = .6;

		double weights1[][] = new double[inputPixels][neurons1];
		double weights2[][] = new double[neurons1][outputs];
		
		//arrays galore
		double[][] inputs = new double[batchSize][inputPixels];
		double[] hiddenSums = new double[neurons1];
		double[] hiddenLayer = new double[neurons1];
		double[] outputSums = new double[outputs];
		double[] outputLayer = new double[outputs];
		int[] labels = new int[batchSize];
		double[] yhat = new double[10];
		double cost;
		
		//reads CSV file into imageInputs array
		Scanner in = new Scanner(new File(System.getProperty("user.dir") + "/mnist_train.csv")); 
		int p;
		for (int image = 0; image < batchSize; image ++)
		{
			String line = in.nextLine();
			String[] lineArray = line.split(",");
			
			labels[image] = Integer.parseInt(lineArray[0]);
			
			p = 1; 
			
			for (int i = 0; i < inputPixels; i++) 
			{
				inputs[image][i] = ((double)Integer.parseInt(lineArray[p])) / 255;
				p++;
			}
		}
		
		//initializes first weights array
		for (int i = 0; i < weights1.length; i++)
		{
			for (int o = 0; o < weights1[i].length; o++)
			{
				weights1[i][o] = Math.random();
				if(Math.random() < .5)
				{
					weights1[i][o] *= -1;
				}
			}
		}
		
		//initializes second weights array
		for (int i = 0; i < weights2.length; i++)
		{
			for (int o = 0; o < weights2[i].length; o++)
			{
				weights2[i][o] = Math.random();
				if(Math.random() < .5)
				{
					weights2[i][o] *= -1;
				}
			}
		}
		
		for(int a = 0; a < batchSize; a++)
		{
			//feed forward hidden layer
			for (int o = 0; o < neurons1; o++)
			{
				for (int i = 0; i < inputPixels; i++)
				{
					hiddenSums[o] += (weights1[i][o] * inputs[a][i]);
				}
				hiddenLayer[o] = -1 / (1 + Math.exp((1/(inputPixels/4.667)) * hiddenSums[o])) + 1;
			}
			
			//feed forward output layer
			for (int o = 0; o < outputs; o++)
			{
				for (int i = 0; i < neurons1; i++)
				{
					outputSums[o] += (weights2[i][o] * hiddenLayer[i]);
				}
				outputLayer[o] = -1 / (1 + Math.exp((1/(neurons1/4.667)) * outputSums[o])) + 1;
			}
			
			//set desired outputs
			for(int i = 0; i < 10; i++)
			{
				if (labels[a] == i)
				{
					yhat[i] = 1.0;
				}
				else
				{
					yhat[i] = 0.0;
				}
			}

			//calculate and print average cost
			cost = 0;
			for(int i = 0; i < 10; i++)
			{
				cost += Math.pow((yhat[i] - outputLayer[i]), 2);
			}
			cost /= 10;
			System.out.println("cost: " + cost);
			
			//backpropagate 2nd weights
			for (int o = 0; o < outputs; o++)
			{
				double n = (yhat[o] - outputLayer[o]); //* sigmoidDev(1/(neurons1/4.667), outputSums[o]);
				for (int i = 0; i < neurons1; i++)
				{
					weights2[i][o] += (n * hiddenLayer[i] * k);
				}
			}
			
			//backpropagate 1st weights
			for (int o = 0; o < neurons1; o++)
			{
				double n = sigmoidDev(1/(inputPixels/4.667), hiddenSums[o]);
				double x = 0;
				for(int oo = 0; oo < outputs; oo++)
				{
					x += ((yhat[oo] - outputLayer[oo]) * -1  * hiddenLayer[o]); //sigmoidDev(1/(neurons1/4.667), outputSums[oo]));
					for(int i = 0; i < inputPixels; i++)
					{
						weights1[i][o] += (n * x * inputs[a][i] * k);
					}
				}
			}
		}
	}
	
	public static double sigmoidDev(double x, double y) 
	{
		return y * Math.exp(x * y) / Math.pow(Math.exp(x * y), 2);
	}
}