#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

// #define EPS 1e-4

using namespace tensorflow;

REGISTER_OP("SortKnot")
	.Input("t: float")
	.Input("p: int32")
	.Output("ts: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
		c->set_output(0, c->input(0));
		return Status::OK();
	});

class SortKnotOp: public OpKernel{
	public:
	explicit SortKnotOp(OpKernelConstruction* context): OpKernel(context){}
	void Compute(OpKernelContext* context) override{
		// Grab the input tensor
		const Tensor& T = context->input(0);
		const Tensor& P = context->input(1);
		auto t = T.vec<float>();
		auto p = P.scalar<int>()(0);
		
		// Create an output tensor
		Tensor* Ts = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, T.shape(), &Ts));
		auto ts = Ts->vec<float>();
		const int knots = t.size();
		
		for(int i = 0; i < knots; i++){
			ts(i) = t(i);
		}
		
		// float bl = t(0);
		// float br = t(knots - 1);
		
		//Fix knots in [t_1, t_{n + p + 1}]]
		float a = t(0);
		float b = t(knots - 1);
		float d = b - a;
		int eta = 0;
		for(int i = p + 1; i < knots - p - 1; i++){
			if(ts(i) > b){
				ts(i) -= a;
				eta = int(ts(i)/d);
				ts(i) = ts(i) - eta*d;
				if(eta%2 == 1){
					ts(i) = d - ts(i);
				}
				ts(i) += a;
			}else if(ts(i) < a){
				ts(i) = b - ts(i);
				eta = int(ts(i)/d);
				ts(i) = ts(i) - eta*d;
				if(eta%2 == 1){
					ts(i) = d - ts(i);
				}
				ts(i) = b - ts(i);
			}
			// ts(i) = fix(ts(i), t(0), t(knots - 1));
		}
		// Sorting interior knots
		quickSort(ts, p + 1, knots - p - 2);
	}
	
	private:
	float fix(float x, float a, float b){
		if(x < a){
			x = 2*a - x;
			return fix(x, a, b);
		}else if(x > b){
			x = 2*b - x;
			return fix(x, a, b);
		}
		return x;
	}
	void quickSort(TTypes<float>::Vec vec, int left, int right){
		int i = left, j = right;
		float pivot = vec((left + right)/2);
		float tmp;
		
		while(i <= j){
			while(vec(i) < pivot)
				i++;
			while(vec(j) > pivot)
				j--;
			if(i <= j){
				tmp = vec(i);
				vec(i) = vec(j);
				vec(j) = tmp;
				i++;
				j--;
			}
		}
		
		if(left < j)
			quickSort(vec, left, j);
		if(i < right)
			quickSort(vec, i, right);
	}
};

REGISTER_KERNEL_BUILDER(Name("SortKnot").Device(DEVICE_CPU), SortKnotOp);