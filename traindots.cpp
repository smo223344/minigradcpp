#include "minigrad.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image_write.h"


Network network;

void save_output(float* output, int x, int y, int iter)
{
	int xx, yy;

	unsigned char* output_bytes = (unsigned char*)malloc(x*y);
	
	// clamp just in case we go above 255 or below 0
	for (xx = 0; xx < x; xx++)
	{
		for (yy = 0; yy < y; yy++)
		{
			int o = (int)((output[yy * x + xx] + 1.0f) * 127.5f);
			if (o < 0)
				o = 0;
			if (o > 255)
				o = 255;
			output_bytes[yy * x + xx] = (unsigned char)o;
		}
	}
	
	char filename[32];
	snprintf(filename, 32, "5test_iter_%03d.png", iter);
	stbi_write_png(filename, x, y, 1, output_bytes, x);

	free(output_bytes);
}



int main(int argc, char** argv)
{
	int x, y, n;
	unsigned char* data = (unsigned char*)malloc(2);
	data[0] = 255;
	data[1] = 0;
	x = 2;
	y = 1;
	n = 1;
	
	printf("x=%d y=%d n=%d\n", x, y, n);

	int i, j;
	for (i = 0; i < 2; i++)
	{
		printf("%02x ", data[i]);
	}

	float* float_data = (float*)malloc(x*y*sizeof(float));

	int xx, yy;
	for (xx = 0; xx < x; xx++)
	{
		for (yy = 0; yy < y; yy++)
		{
			float_data[yy * x + xx] = (float)data[yy * x + xx] / 127.5f - 1.0f;
		}
	}

	network.init(2);
	

	int total_epochs = 100;
	int epochs = total_epochs;
	float coords[3] = { 0.0f, 0.0f, 0.0f };
	float encoded_input[20];
	float* output = (float*)malloc(x*y*sizeof(float));
//	float learning_rate = (1.0f / 255.0f) * 0.5f;
	float learning_rate = 0.001f;
	while (epochs--)
	{
		if (epochs == 800) learning_rate /= 5.0f;
		if (epochs == 650) learning_rate /= 5.0f;
		if (epochs == 450) learning_rate /= 5.0f;
		if (epochs == 400) learning_rate /= 5.0f;
		if (epochs == 300) learning_rate /= 5.0f;
		if (epochs == 200) learning_rate /= 5.0f;
		if (epochs == 100) learning_rate /= 5.0f;
		if (epochs == 70) learning_rate /= 5.0f;
		if (epochs == 35) learning_rate /= 5.0f;
		if (epochs == 10) learning_rate /= 2.0f;

		float cumulative_loss = 0.0f;
//		for (yy = 30; yy < 70; yy++)
		int test_count = 0;
		for (; test_count < 100; test_count++)
		{
//			// We want to randomize the order in which lines are trained because the smoothness of moving continously from one row to the next is killing the loss and making learning difficult.
//			// TODO shift register
			xx = test_count % 2;
			yy = 0;
			coords[0] = yy / 50.0f - 1.0f;
			coords[1] = xx / 50.0f - 1.0f;
//			coords[0] = (30 + rand() % 40) / 50.0f - 1.0f;
//			Network::positional_encode(coords[0], encoded_input, 10);
//			Network::positional_encode(coords[1], &encoded_input[10], 10);	
			encoded_input[0] = xx; encoded_input[1] = yy;

			float loss = network.stochastic_fit(encoded_input, 2, &float_data[yy * x + xx], learning_rate, &output[yy * x + xx]);
//			printf("yy=%d epoch=%d loss=%f\n", yy, epochs, loss);
			cumulative_loss += loss;
		}
/*		float new_learning_rate = cumulative_loss / 40.0f / 10000.0f;
		if (new_learning_rate < learning_rate)
			learning_rate = new_learning_rate;*/
		printf("epoch = %d, avg loss = %f\n", epochs, cumulative_loss / test_count);
/*
		if (epochs % 5 == 0)
		{
			yy = 0;
				for (xx = 0; xx < 2; xx++)
				{
					// infer 
					coords[0] = yy / 50.0f - 1.0f;
					coords[1] = xx / 50.0f - 1.0f;
					Network::positional_encode(coords[0], encoded_input, 10);
					Network::positional_encode(coords[1], &encoded_input[10], 10);
					float loss = network.stochastic_fit(encoded_input, 20, &float_data[yy * x + xx], 0, &output[yy * x + xx]);
				}
			save_output(output, x, y, total_epochs - epochs);
		}*/
	}
	
	
	yy = 0;
	{
		for (xx = 0; xx < 2; xx++)
		{
			// infer 
			coords[0] = yy / 50.0f - 1.0f;
			coords[1] = xx / 50.0f - 1.0f;
//			Network::positional_encode(coords[0], encoded_input, 10);
//			Network::positional_encode(coords[1], &encoded_input[10], 10);
			encoded_input[0] = xx;
			encoded_input[1] = yy;
			float loss = network.stochastic_fit(encoded_input, 2, &float_data[yy * x + xx], 0, &output[yy * x + xx]);
		}
	}

	save_output(output, x, y, 99999);

	return 0;
}
