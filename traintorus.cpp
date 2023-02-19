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

	int epochs = 400;
	float coords[3] = { 0.0f, 0.0f, 0.0f };
	float* output = (float*)malloc(x*y*sizeof(float));
//	float learning_rate = (1.0f / 255.0f) * 0.5f;
	float learning_rate = 0.01f;
	while (epochs--)
	{
		if (epochs == 200) learning_rate /= 10.0f;
		if (epochs == 100) learning_rate /= 10.0f;
		if (epochs == 50) learning_rate /= 10.0f;

		float cumulative_loss = 0.0f;
//		for (yy = 20; yy < 60; yy++)
		int test_count = 0;
		for (; test_count < 40; test_count++)
		{
			yy = 30 + (rand() % 40);
			coords[0] = yy -50.0f;/// 200.0f - 100.0f;

			float loss = network[/*yy*/0].stochastic_fit(&coords[0], 1, &float_data[yy * x], learning_rate, &output[yy * x]);
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
				o = o;
			if (o > 255)
				o = 255;
			output_bytes[yy * x + xx] = (unsigned char)o;
		}
	}
	
	stbi_write_png("test_1_pre_infer.png", x, y, 1, output_bytes, x);
	
	for (yy = 0; yy < 100; yy++)
	{
		// TODO infer here
		coords[0] = yy-50.0f;// / 200.0f - 100.0f;
		float loss = network[0].stochastic_fit(&coords[0], 3, &float_data[yy * x], 0, &output[yy * x]);
	}

	for (xx = 0; xx < x; xx++)
	{
		for (yy = 0; yy < y; yy++)
		{
			int o = (int)((output[yy * x + xx] + 1.0f) * 127.5f);
			if (o < 0)
				o = o;
			if (o > 255)
				o = 255;
			output_bytes[yy * x + xx] = (unsigned char)o;
		}
	}

	stbi_write_png("test_2_post_infer.png", x, y, 1, output_bytes, x);


	return 0;
}
