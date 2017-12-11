#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#define EPS 0

using namespace tensorflow;

REGISTER_OP("FixedBSplineGrad")
	.Input("g: float")
	.Input("x: float")
	.Input("t: float")
	.Input("c: float")
	.Input("p: int32")
	.Output("gx: float")
	.Output("gc: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
		c->set_output(0, c->input(1));
		c->set_output(1, c->input(3));
		return Status::OK();
	});
	
class FixedBSplineGradOp: public OpKernel{
	public:
	explicit FixedBSplineGradOp(OpKernelConstruction* context): OpKernel(context){}
	void Compute(OpKernelContext* context) override{
		// Grab the input tensor
		const Tensor& G = context->input(0);
		const Tensor& X = context->input(1);
		const Tensor& T = context->input(2);
		const Tensor& C = context->input(3);
		const Tensor& P = context->input(4);
		auto g = G.matrix<float>();
		auto x = X.matrix<float>();
		auto t = T.vec<float>();
		auto c = C.matrix<float>();
		auto p = P.scalar<int>()(0);

		// Create an output tensor
		Tensor* Gx = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, X.shape(), &Gx));
		auto gx = Gx->matrix<float>();
		Tensor* Gc = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(1, C.shape(), &Gc));
		auto gc = Gc->matrix<float>();
		
		const int internalKnots = C.shape().dim_size(1);
		const int knots = t.size();
		const int samples = Gx->shape().dim_size(0);
		const int nodes = Gx->shape().dim_size(1);
		
		int* head = new int[samples*nodes];
		int** xmu = new int*[nodes];
		for(int i = 0; i < nodes; i++){
			xmu[i] = &head[i*samples];
			for(int r = 0; r < samples; r++){
				xmu[i][r] = binarySearch(x(r, i), t);
			}
		}
		
		//Zero the c grad, as it is accumulated by a sum later
		for(int i = 0; i < nodes; i++){
			for(int j = 0; j < internalKnots; j++){
				gc(i, j) = 0;
			}
		}

		//No more than 100 knots are supported currently
		//dynamic arrays gives errors for some reason,
		//multithreading in tensorflow probably
		
		float d[100];
		int mu;
		float a;
		float xx;
		for(int i = 0; i < nodes; i++){
			for(int r = 0; r < samples; r++){
				xx = x(r, i);
				if(xx < t(p) || xx >= t(knots - p - 1)){
					gx(r, i) = 0;
					continue;
				}
				mu = xmu[i][r];
				for(int j = mu - p; j <= mu; j++){
					d[j] = c(i, j);
				}
				//Stopping at k = 2 in order to perform differentiation on last step
				for(int k = p; k > 1; k--){
					for(int j = mu; j > mu - k; j--){
						a = (xx - t(j))/(t(j + k) - t(j));
						d[j] = (1 - a)*d[j - 1] + a*d[j];
					}
				}
				//Derivative procedure
				gx(r, i) = g(r, i)*p*(d[mu] - d[mu - 1])/(t(mu + 1) - t(mu));
			}
		}
	
		//Left to right algorithm
		for(int i = 0; i < nodes; i++){
			for(int r = 0; r < samples; r++){
				xx = x(r, i);
				if(xx < t(p)){
					gc(i, 0) += g(r, i);
					continue;
				}
				if(xx >= t(knots - p - 1)){
					gc(i, internalKnots - 1) += g(r, i);
					continue;
				}
				mu = xmu[i][r];
				a = (xx - t(mu))/(t(mu + 1) - t(mu));
				d[mu - 1] = 1 - a;
				d[mu] = a;
				for(int k = 2; k <= p; k++){
					a = (xx - t(mu + 1 - k))/(t(mu + 1) - t(mu + 1 - k));
					d[mu - k] = (1 - a)*d[mu - k + 1];
					for(int j = k - 1; j > 0; j--){
						d[mu - j] = a*d[mu - j];
						a = (xx - t(mu - j + 1))/(t(mu + k - j + 1) - t(mu - j + 1));
						d[mu - j] += (1 - a)*d[mu - j + 1];
					}
					d[mu] = a*d[mu];
				}
				for(int l = mu - p; l <= mu; l++){
					gc(i, l) += g(r, i)*d[l];
				}
				
			}
		}
		delete[] head;
	}
	
	private:
	int binarySearch(float x, TTypes<const float>::Vec t){
		int l = 0;
		int r = t.size() - 1;
		int m;
		
		while(r - l > 1){
			m = (l + r)/2;
			if(t(m) <= x){
				l = m;
			}else{
				r = m;
			}
		}
		return l;
	}
};

REGISTER_KERNEL_BUILDER(Name("FixedBSplineGrad").Device(DEVICE_CPU), FixedBSplineGradOp);