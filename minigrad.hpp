#include <vector>
#include <set>
#include <ranges>

#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdio>

#define LW 64
#define LH 3
#define LAYER_TYPES	'r', 'r', 0
#define LWO 1

#define ACTIVATION_SIGMOID 0
#define ACTIVATION_RELU 1

struct Value
{
	float value;
	char op;

//	std::vector<Value*> prev;
	Value* prev0;
	Value* prev1;

	float p1, p2;

	float grad;

	static float rand_normal()
	{
		int i;
		float sum = 0.0f;

		for (i = 0; i < 10; i++)
		{
			sum += ((rand() % 2000) / 1000.0f - 1.0f) / 1000.0f;
		}

//		return fabs(sum);
		return sum;
	}
	

	Value(char _op, float a1, float a2, Value* arg1, Value* arg2)
	{
		grad = 0.0;
		op = _op;
		prev0 = NULL;
		prev1 = NULL;


		if (arg1)
		{
			a1 = arg1->value;
			prev0 = arg1;
		}
		p1 = a1;

		if (arg2)
		{
			a2 = arg2->value;
			prev1 = arg2;
		}
		p2 = a2;
		
		
		switch (op)
		{
			case 0:
				value = a1;
				break;
			case '+':
				value = a1 + a2;
				break;

//			case '-':
//				value = a1 - a2;
//				break;

			case '*':
				value = a1 * a2;
				break;

//			case '/':
//				value = a1 / a2;
//				break;

			case '^':
				value = powf(a1, a2);
				break;

			case 'r':
				value = (a1 > 0.0f) ? a1 : 0.0f;
				break;

			case 't':
				value = tanhf(a1);
				break;

			default:
				printf("invalid value op %c\n", op);
				assert(0);
		}
	}

	Value(float v) : Value(0, v, 0, 0, 0)
	{
	
	}

	~Value()
	{
//		prev.clear();
	}

	void backward_op()
	{
/*		float p1, p2;

		if (prev0)
			p1 = prev0->value;
		else
			p1 = a1;

		if (prev1)
			p2 = prev1->value;
		else
			p2 = a2;
*/
//		if (grad > 0.0f)
//		printf("%c %f\n", op ? op : '0', grad);
//
/*
		if (grad > 10000.0f)
			grad = 10000.0f;
		if (grad < -10000.0f)
			grad = -10000.0f;*/
		switch (op)
		{
			case 0:
				// leaf
//				printf("grad of leaf is %f\n", grad); 
				break;

			case '+':
				if (prev0) prev0->grad += grad;
				if (prev1) prev1->grad += grad;
				break;

			case '*':
				if (prev0) prev0->grad += p2 * grad;
				if (prev1) prev1->grad += p1 * grad;
				break;

			case '^':
				if (prev0) prev0->grad += (p2 * powf(prev0->value, p2 - 1.0f)) * grad;
				break;

			case 'r':
				if (prev0) prev0->grad += ((p1 > 0.0f) ? 1.0f : 0.00001f) * grad;
				break;

			case 't':
//				printf("original prev grad: %f\n", prev0->grad);
//				printf("p1=%f\n", p1);
//				printf("dTanh() = %f\n", (1.0f - tanhf(p1) * tanhf(p1)));
				if (prev0) 
				{
					float dtanh = (1.0f - tanhf(p1) * tanhf(p1));
					if (fabs(dtanh) < 0.01f)
					{
						dtanh = p1/2.0f;
					}
					prev0->grad = dtanh * grad;
				}
//				printf("tanh grad: %f tanh prev grad: %f\n", grad, prev0->grad);
				break;

			default:
				printf("invalid backward_op %c\n", op);
				assert(0);

		}
	}

	static void build_topo(std::set<Value*>& visited, std::vector<Value*>& topo, Value* v)
	{
		if (!visited.count(v))
		{
			visited.insert(v);
//			for (auto&& c : v->prev)
//			{
//				build_topo(visited, topo, c);
//			}
			if (v->prev0) build_topo(visited, topo, v->prev0);
			if (v->prev1) build_topo(visited, topo, v->prev1);			
			topo.push_back(v);
		}
	}

#define VALUE_COUNT (LH * (LW * (LW + 1 + 4 * LW)) + 4 * LW)


	void backward(std::vector<Value*>& topo)
	{
		std::set<Value*> visited;

		if (topo.empty())
		{
			topo.reserve(VALUE_COUNT);
			build_topo(visited, topo, this);
		}

		grad = 1.0f;

		assert(topo.back()->grad == 1.0f);

/*		while (!topo.empty())
		{
			Value* v = topo.back(); topo.pop_back();

			v->backward_op();
		}
*/
		for (auto&& v: topo | std::views::reverse)
		{
//printf("b: %f\n", v->grad);
			v->backward_op();
		}
	}

	void zero_grad(std::vector<Value*>& topo)
	{
		std::set<Value*> visited;

		if (topo.empty())
		{
			topo.reserve(VALUE_COUNT);
			build_topo(visited, topo, this);
		}

//printf("zeroing %d...\n", topo.size());
		for (auto&& v : topo)
		{
			v->grad = 0.0;
		}
	}

/*
	Value* do_op(Value* arg1, Value*)
	{
		arg = a;
		switch (op)
		{
			case 0: break;
			case '+':
				out.init(a, '+')
				value + a;
			case '*':
				return value * a;
			case 'r':
				if (a > 0.0)
					return a;
				return 0.01 * a;
			default:
				printf("invalid op\n");

		}

		return 0;
	}
*/
};

struct Node
{
	Value* bias;
	std::vector<Value> xfer_terms;
	std::vector<Value> xfer_sums;
	Value* result;
	char act_op;
	int input_width;
	Value* weights[LW]; // maximum number of weights a node can have is the maximum layer width in general.
	
	Node() : bias(NULL), result(NULL), input_width(0) {}

	~Node()
	{
		delete result;
		delete bias;
		int i;
		for (i = 0; i < input_width; i++)
		{
			delete weights[i];
		}
	}

	void init(char act_op, int input_width)
	{
		assert(input_width <= LW);

		this->input_width = input_width;
		bias = new Value(Value::rand_normal());
//		bias = new Value((rand() % 20000 - 10000) / 10000.0f);
//		bias = new Value(0.0f);
		this->act_op = act_op;

		int i;
		for (i = 0; i < input_width; i++)
		{
			weights[i] = new Value(Value::rand_normal());
//			weights[i] = new Value((rand() % 20000 - 10000) / 10000.0f);
//			weights[i] = new Value((rand() % 20000 - 10000) / 10000.0f);
//			weights[i] = new Value(1.0f);
//			printf("w=%f\n", weights[i]->value);
		}
	}

	void collect_params(std::vector<Value*>& params)
	{
		params.push_back(bias);
		int i;
		for (i = 0; i < input_width; i++)
		{
			params.push_back(weights[i]);
		}
	}

	Value* forward(Value** x)
	{
		int i;

		if (result)
			free(result);

		// Reserve capacity in vectors so pointers aren't invalidated by emplace_back() calls
		xfer_terms.clear();
		xfer_terms.reserve(input_width + 2);

		xfer_sums.clear();
		xfer_sums.reserve(input_width + 2);

		// Multiply all xi*wi and place into xfer_terms (transfer function terms)
		for (i = 0; i < input_width; i++)
		{
			xfer_terms.emplace_back('*', 0.0, 0.0, x[i], weights[i]);
		}

		// Add in the bias and build a sum of xfer terms. Skip last term to return as result if activation is linear.
		xfer_sums.emplace_back('+', 0.0, 0.0, bias, &xfer_terms[0]);
		for (i = 1; act_op ? (i < input_width) : (i < input_width - 1); i++)
		{
			xfer_sums.emplace_back('+', 0.0, 0.0, &xfer_sums[xfer_sums.size() - 1], &xfer_terms[i]);
		}
		
		if (act_op == 0)
		{
			// linear pass through -- just add the last transfer function term
			result = new Value('+', 0.0, 0.0, &xfer_sums[xfer_sums.size()-1], &xfer_terms[input_width-1]);
		} else
		if (act_op == 't')
		{
			result = new Value('t', 0.0, 0.0, &xfer_sums[xfer_sums.size()-1], (Value*)NULL);
		} else
		if (act_op == 'r')
		{
			// relu -- arg is final transfer function sum
			result = new Value('r', 0.0, 0.0, &xfer_sums[xfer_sums.size()-1], (Value*)NULL);
		} else
		{
			printf("invalid act op %c\n", act_op);
			assert(0);
		}

		return result;
	}



};

struct Layer
{
	int layer_width;
	int input_width;
	Node nodes[LW];

	void init(char act_op, int layer_width, int input_width)
	{
		assert(layer_width <= LW);
		
		this->input_width = input_width;
		this->layer_width = layer_width;

		int i;
		for (i = 0; i < layer_width; i++)
		{
			nodes[i].init(act_op, input_width);
		}
	}

	void collect_params(std::vector<Value*>& params)
	{
		int i;
		for (i = 0; i < layer_width; i++)
		{
			nodes[i].collect_params(params);
		}
	}

	void forward(Value** inputs, Value** outputs)
	{
		int i;
		for (i = 0; i < layer_width; i++)
		{
			outputs[i] = nodes[i].forward(inputs);	
		}
	}
};


class Network
{
	public:
	Layer layers[LH];

	std::vector<Value*> params;

	Network()
	{
		params.clear();
	}

	int input_width;
	int last_layer_width;

	void init(int input_width)
	{
		this->input_width = input_width;
		last_layer_width = LWO;

		params.clear();
		params.reserve((LH + 1) * (LW * (LW + 1)));

/*		char layer_types[LH] = {0, 0, 0, 'r', 
					0, 'r', 0, 'r',
					0, 0, 0, 'r',
					0, 'r', 0, 't' 
					};*/
		char layer_types[LH] = {
					LAYER_TYPES
					};
	
		int i;
		for (i = 0; i < LH; i++)
		{
			layers[i].init(
//					(i < LH - 1) ? 0 : 'r', // last layer is relu, for pixels, for some reason.
					layer_types[i],
					(i == LH - 1) ? LWO : LW, // last layer has LWO nodes, everything else has LW
					(i == 0) ? input_width : LW // first layer has input_width inputs, everything else has LW inputs
			);

			layers[i].collect_params(params);
		}
		
		memset(activations, 0, sizeof(activations));
	}

	Value** input_values;
	Value** output_values;
	Value* activations[LH+1][LW];

	void forward(float* input, int input_width, float* output)
	{
		assert(input_width == this->input_width);

		int i;

		// set up the inputs		
		for (i = 0; i < input_width; i++)
		{
			if (activations[0][i])
				delete activations[0][i];
			activations[0][i] = new Value(input[i]);
		}
		input_values = activations[0];

		// feed forward
		for (i = 0; i < LH; i++)
		{
			layers[i].forward(activations[i], activations[i+1]);
		}

		output_values = activations[LH];
		for (i = 0; i < last_layer_width; i++)
		{
			output[i] = activations[LH][i]->value;
		}
	}

	std::vector<Value> loss_scratch;
	Value* compute_loss(float* predicted)
	{
		loss_scratch.clear();
		loss_scratch.reserve(4 * last_layer_width);

		// loss = sum ( (out - pred)^2 )
		int i;
		for (i = 0; i < last_layer_width; i++)
		{
//printf("out=%f pred=%f\n", output_values[i]->value, predicted[i]);
			loss_scratch.emplace_back('+', 0.0, 0.0f-predicted[i], output_values[i], (Value*)NULL);
			loss_scratch.emplace_back('+', 0.0, 0.0f-predicted[i], output_values[i], (Value*)NULL);
			loss_scratch.emplace_back('*', 0.0, 0.0, &loss_scratch[loss_scratch.size() - 1], &loss_scratch[loss_scratch.size() - 2]);
		}
		if (last_layer_width > 1)
		{
			loss_scratch.emplace_back('+', 0.0, 0.0, &loss_scratch[0*3 + 2], &loss_scratch[1*3 + 2]);
			for (i = 2; i < last_layer_width; i++)
			{
				loss_scratch.emplace_back('+', 0.0, 0.0, &loss_scratch[i*3 + 2], &loss_scratch[loss_scratch.size() - 1]);
			}
		}

		return &loss_scratch[loss_scratch.size() - 1];
	}

	void optimize(float learning_rate)
	{
		int nonzero_count = 0;
		for (auto&& v : params)
		{
//printf("%f - %f * %f =>", v->value, learning_rate, v->grad);
			
			v->value = v->value - learning_rate * v->grad;
//			if (fabs(v->grad) > 0.0000000001f)
//			{
//				nonzero_count++;
//			}

//printf("%f\n", v->value);
		}

		// TODO warn if zero grads?

//		printf("optimized %d / %d nodes\n", nonzero_count, params.size());
	}

	void zero_grad()
	{
		printf("zero_grad() the loss instead, it'll be faster.\n");
		assert(0);
		/*
		int i;
		for (i = 0; i < LW; i++)
		{

		}*/
	}

	float stochastic_fit(float* input, int input_width, float* predicted, float learning_rate, float* output)
	{
//		float* output = new float[LW];

		forward(input, input_width, output);
		Value* loss = compute_loss(predicted);
		if (learning_rate > 0.0f)
		{
			std::vector<Value*> topo;
			loss->zero_grad(topo);
			loss->backward(topo);
			optimize(learning_rate);
		}
		
//		delete[] output;
		return loss->value;		
	}

	static void positional_encode(double x, float* encoded, int n_encoded)
	{
		int i;
		double coefficient = 2.0;
		for (i = 0; i < n_encoded; i++)
		{
			if (i % 2)
			{
				encoded[i] = (float)(M_PI * cos((double)x) * coefficient);
				coefficient *= 2.0;
			} else
			{
				encoded[i] = (float)(M_PI * sin((double)x) * coefficient);
			}
		}
	}
};
