#include "minigrad.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image_write.h"


Network network[100];

int main(int argc, char** argv)
{
	int x, y, n;
	unsigned char* data = stbi_load("../../t/1.png", &x, &y, &n, 1);
	
	printf("x=%d y=%d n=%d\n", x, y, n);

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
		network[yy].init();
	}

	int epochs = 120;
	float coords[3] = { 0.0f, 0.0f, 0.0f };
	float encoded_input[10];
	float* output = (float*)malloc(x*y*sizeof(float));
//	float learning_rate = (1.0f / 255.0f) * 0.5f;
	float learning_rate = 0.01f;
	while (epochs--)
	{
		if (epochs == 600) learning_rate /= 2.0f;
		if (epochs == 500) learning_rate /= 2.0f;
		if (epochs == 450) learning_rate /= 2.0f;
		if (epochs == 400) learning_rate /= 2.0f;
		if (epochs == 300) learning_rate /= 2.0f;
		if (epochs == 200) learning_rate /= 2.0f;
		if (epochs == 100) learning_rate /= 2.0f;
		if (epochs == 70) learning_rate /= 2.0f;
		if (epochs == 35) learning_rate /= 2.0f;
		if (epochs == 10) learning_rate /= 2.0f;

		float cumulative_loss = 0.0f;
		for (yy = 30; yy < 70; yy++)
//		int test_count = 0;
//		for (; test_count < 40; test_count++)
		{
//			yy = 30 + (rand() % 40);
//			coords[0] = yy / 50.0f - 1.0f;
//			// We want to randomize the order in which lines are trained because the smoothness of moving continously from one row to the next is killing the loss and making learning difficult.
			coords[0] = (30 + rand() % 40) / 50.0f - 1.0f;
			Network::positional_encode(coords[0], encoded_input, 10);

			float loss = network[0].stochastic_fit(encoded_input, 10, &float_data[yy * x], learning_rate, &output[yy * x]);
//			printf("yy=%d epoch=%d loss=%f\n", yy, epochs, loss);
			cumulative_loss += loss;
		}
/*		float new_learning_rate = cumulative_loss / 40.0f / 10000.0f;
		if (new_learning_rate < learning_rate)
			learning_rate = new_learning_rate;*/
		printf("epoch = %d, avg loss = %f\n", epochs, cumulative_loss / 40.0f);
	}
	
	unsigned char* output_bytes = (unsigned char*)malloc(x*y);
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
	
	stbi_write_png("2test_1_pre_infer.png", x, y, 1, output_bytes, x);
	
	for (yy = 30; yy < 70; yy++)
	{
		// TODO infer here
		coords[0] = yy / 50.0f - 1.0f;
		Network::positional_encode(coords[0], encoded_input, 10);
		float loss = network[0].stochastic_fit(encoded_input, 10, &float_data[yy * x], 0, &output[yy * x]);
	}

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

	stbi_write_png("2test_2_post_infer.png", x, y, 1, output_bytes, x);


	return 0;
}
