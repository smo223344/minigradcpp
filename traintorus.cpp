#include "minigrad.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image_write.h"


Network network[100];

int main(int argc, char** argv)
{
	int x, y, n;
	unsigned char* data = stbi_load("../t/1.png", &x, &y, &n, 1);
	
	printf("x=%d y=%d n=%d\n", x, y, n);

	float* float_data = (float*)malloc(x*y*sizeof(float));

	int xx, yy;
	for (xx = 0; xx < x; xx++)
	{
		for (yy = 0; yy < y; yy++)
		{
			float_data[yy * x + xx] = ((float)data[yy * x + xx]) / 127.5f - 1.0f;
		}
	}

	for (yy = 0; yy < 100; yy++)
	{
		network[yy].init();
	}

	int epochs = 200;
	float coords[2] = { 1.0f, 1.0f };
	float* output = (float*)malloc(x*y*sizeof(float));
	float learning_rate = (1.0f / 255.0f) * 0.1f;
	while (epochs--)
	{
		float cumulative_loss = 0.0f;
		for (yy = 20; yy < 60; yy++)
		{
			float loss = network[yy].stochastic_fit(&coords[0], 2, &float_data[yy * x], learning_rate, &output[yy * x]);
			printf("yy=%d epoch=%d loss=%f\n", yy, epochs, loss);
			cumulative_loss += loss;
		}
/*		float new_learning_rate = cumulative_loss / 40.0f / 10000.0f;
		if (new_learning_rate < learning_rate)
			learning_rate = new_learning_rate;*/
	}

	unsigned char* output_bytes = (unsigned char*)malloc(x*y);
	for (xx = 0; xx < x; xx++)
	{
		for (yy = 0; yy < y; yy++)
		{
			output_bytes[yy * x + xx] = (unsigned char)((output[yy * x + xx] + 1.0f)*127.5f);
		}
		stbi_write_png("test.png", x, y, 1, output_bytes, x);
	}


	return 0;
}
