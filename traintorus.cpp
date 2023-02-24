#include "minigrad.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image_write.h"


Network network[100];

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
	snprintf(filename, 32, "2test_iter_%03d.png", iter);
	stbi_write_png(filename, x, y, 1, output_bytes, x);

	free(output_bytes);
}



int main(int argc, char** argv)
{
	int x, y, n;
	unsigned char* data = stbi_load("../../t/2.png", &x, &y, &n, 1);
	
	printf("x=%d y=%d n=%d\n", x, y, n);

	int i, j;
	for (i = 0; i < 100; i++)
	{
		if (i % 3 == 0)
		{
			for (j = 0; j < 100; j++)
			{
				if (j % 3 == 0)
					printf("%02x ", data[i * x + j]);
			}
			printf("\n");
		}
		
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

	for (yy = 0; yy < 100; yy++)
	{
		network[yy].init(10);
	}

	int total_epochs = 1000;
	int epochs = total_epochs;
	float coords[3] = { 0.0f, 0.0f, 0.0f };
	float encoded_input[10];
	float* output = (float*)malloc(x*y*sizeof(float));
//	float learning_rate = (1.0f / 255.0f) * 0.5f;
	float learning_rate = 0.001f;
	while (epochs--)
	{
		if (epochs == 600) learning_rate /= 10.0f;
		if (epochs == 500) learning_rate /= 5.0f;
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
		for (; test_count < 40; test_count++)
		{
//			// We want to randomize the order in which lines are trained because the smoothness of moving continously from one row to the next is killing the loss and making learning difficult.
			yy = 30 + (rand() % 40);
			coords[0] = yy / 50.0f - 1.0f;
//			coords[0] = (30 + rand() % 40) / 50.0f - 1.0f;
			Network::positional_encode(coords[0], encoded_input, 10);

			float loss = network[0].stochastic_fit(encoded_input, 10, &float_data[yy * x], learning_rate, &output[yy * x]);
//			printf("yy=%d epoch=%d loss=%f\n", yy, epochs, loss);
			cumulative_loss += loss;
		}
/*		float new_learning_rate = cumulative_loss / 40.0f / 10000.0f;
		if (new_learning_rate < learning_rate)
			learning_rate = new_learning_rate;*/
		printf("epoch = %d, avg loss = %f\n", epochs, cumulative_loss / 40.0f);

		if (epochs % 50 == 0)
		{
			// TODO Note we are saving the buffer of different training passes combined so it won't look quite right
			save_output(output, x, y, total_epochs - epochs);
		}
	}
	
	
	for (yy = 30; yy < 70; yy++)
	{
		// infer 
		coords[0] = yy / 50.0f - 1.0f;
		Network::positional_encode(coords[0], encoded_input, 10);
		float loss = network[0].stochastic_fit(encoded_input, 10, &float_data[yy * x], 0, &output[yy * x]);
	}

	save_output(output, x, y, 99999);

	return 0;
}
