#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("SortKnotGrad")
	.Input("g: float")
	.Input("t: float")
	.Input("p: int32")
	.Output("gs: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
		c->set_output(0, c->input(0));
		return Status::OK();
	});

class SortKnotGradOp: public OpKernel{
	public:
	explicit SortKnotGradOp(OpKernelConstruction* context): OpKernel(context){}
	void Compute(OpKernelContext* context) override{
		const Tensor& G = context->input(0);
		const Tensor& T = context->input(1);
		const Tensor& P = context->input(2);
		auto g = G.vec<float>();
		auto t = T.vec<float>();
		auto p = P.scalar<int>()(0);
		
		Tensor* Ga = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, G.shape(), &Ga));
		auto ga = Ga->vec<float>();
		
		const int knots = g.size();
		
		int* arg = new int[knots];
		float* tf = new float[knots];
		float* gf = new float[knots];
		for(int i = 0; i < knots; i++){
			tf[i] = t(i);
			arg[i] = i;
		}
		for(int i = 0; i <= p; i++){
			gf[i] = 0;
			gf[knots - i - 1] = 0;
		}
		// Derivative of fixing knots in [t_1, t_{n + p + 1}]]
		int q = 1;
		float a = t(0);
		float b = t(knots - 1);
		float d = b - a;
		int eta = 0;
		for(int i = p + 1; i < knots - p - 1; i++){
			if(tf[i] > b){
				tf[i] -= a;
				eta = int(tf[i]/d);
				tf[i] = tf[i] - eta*d;
				if(eta%2 == 1){
					tf[i] = d - tf[i];
					gf[i] = -g(i);
				}else{
					gf[i] = g(i);
				}
				tf[i] += a;
			}else if(tf[i] < a){
				tf[i] = b - tf[i];
				eta = int(tf[i]/d);
				tf[i] = tf[i] - eta*d;
				if(eta%2 == 1){
					tf[i] = d - tf[i];
					gf[i] = -g(i);
				}else{
					gf[i] = g(i);
				}
				tf[i] = b - tf[i];
			}else{
				gf[i] = g(i);
			}
		}
		// for(int i = p + 1; i < knots - p - 1; i++){
		// 	tf[i] = fix(tf[i], &q, tf[0], tf[knots - 1]);
		// 	gf[i] = q*g(i);
		// 	q = 1;
		// }
		
		//Derivative of sorting knots
		quickArgSort(arg, tf, p + 1, knots - p - 2);
		for(int i = 0; i < knots; i++){
			ga(i) = gf[arg[i]];
		}
		delete[] tf;
		delete[] gf;
		delete[] arg;
	}
	private:
	float fix(float x, int* q, float a, float b){
		if(x < a){
			x = 2*a - x;
			*q = -*q;
			return fix(x, q, a, b);
		}else if(x > b){
			x = 2*b - x;
			*q = -*q;
			return fix(x, q, a, b);
		}
		return x;
	}
	void quickArgSort(int* arg, float* vec, int left, int right){
		int i = left, j = right;
		float pivot = vec[arg[(left + right)/2]];
		int tmp;
		
		while(i <= j){
			while(vec[arg[i]] < pivot)
				i++;
			while(vec[arg[j]] > pivot)
				j--;
			if(i <= j){
				tmp = arg[i];
				arg[i] = arg[j];
				arg[j] = tmp;
				i++;
				j--;
			}
		}
		
		if(left < j)
			quickArgSort(arg, vec, left, j);
		if(i < right)
			quickArgSort(arg, vec, i, right);
	}
};

REGISTER_KERNEL_BUILDER(Name("SortKnotGrad").Device(DEVICE_CPU), SortKnotGradOp);