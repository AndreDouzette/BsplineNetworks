#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#define EPS 0

using namespace tensorflow;

REGISTER_OP("BSplineGrad")
	.Input("g: float")
	.Input("x: float")
	.Input("t: float")
	.Input("c: float")
	.Input("p: int32")
	.Output("gx: float")
	.Output("gt: float")
	.Output("gc: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
		c->set_output(0, c->input(1));
		c->set_output(1, c->input(2));
		c->set_output(2, c->input(3));
		return Status::OK();
	});
	
class BSplineGradOp: public OpKernel{
	public:
	explicit BSplineGradOp(OpKernelConstruction* context): OpKernel(context){}
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
		Tensor* Gt = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(1, T.shape(), &Gt));
		auto gt = Gt->vec<float>();
		Tensor* Gc = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(2, C.shape(), &Gc));
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
		
		// Knots of size<100 supported
		float tl[100];
		float dd[100];
		int mi, ma;
		float tmi, tma;
		//Zero the t grad as it is accumulated by a sum later
		//Initializing extended knot vector
		for(int i = 0; i < knots; i++){
			gt(i) = 0;
			tl[i + 1] = t(i);
		}
		tl[0] = t(0);
		
		//Only the interior knots are free
		for(int l = p + 1; l < knots - p - 1; l++){
			tl[l] = tl[l + 1];
			//Values used to find B-spline coefficients and support interval
			if(l >= internalKnots){
				ma = internalKnots;
				mi = l - p;
				tmi = t(l - p);
				tma = t(internalKnots + p);
			}else if(l < p + 1){
				mi = 1;
				ma = l + 1;
				tmi = t(0);
				tma = t(l + p);
			}else{
				ma = l + 1;
				mi = l - p;
				tmi = t(l - p);
				tma = t(l + p);
			}
			for(int j = 0; j < mi; j++){
				dd[j] = 0;
			}
			for(int j = ma; j < internalKnots + 1; j++){
				dd[j] = 0;
			}
			for(int i = 0; i < nodes; i++){
				for(int j = mi; j < ma; j++){
					if(tl[j + p + 1] - tl[j] < EPS){
						dd[j] = 0;
					}else{
						dd[j] = (c(i, j - 1) - c(i, j))/(tl[j + p + 1] - tl[j]);
					}
				}
				for(int r = 0; r < samples; r++){
					xx = x(r, i);
					if(xx < tmi || xx >= tma){
						continue;
					}
					mu = xmu[i][r];
					if(l <= mu){
						mu += 1;
					}
					for(int j = mu - p; j <= mu; j++){
						d[j] = dd[j];
					}
					for(int k = p; k > 0; k--){
						for(int j = mu; j > mu - k; j--){
							a = tl[j + k] - tl[j];
							a = (xx - tl[j])/(tl[j + k] - tl[j]);
							d[j] = d[j - 1] + a*(d[j] - d[j - 1]);
						}
					}
					if(d[mu] > 1e5 || d[mu] < -1e5 || d[mu] != d[mu]){
						std::cout << mu << ", " << mu << ", " << d[mu] << "\n";
						std::cout << tl[mu] << " < " << xx << " < " << tl[mu + 1] << "\n";
						std::cout << "d:  ";
						for(int q = 0; q < internalKnots; q++){
							std::cout << d[q] << " ";
						}
						std::cout << "\n";
						std::cout << "dd: ";
						for(int q = 0; q < internalKnots; q++){
							std::cout << dd[q] << " ";
						}
						std::cout << "\n";
						std::cout << "tl: ";
						for(int q = 0; q < knots + 1; q++){
							std::cout << tl[q] << " ";
						}
						std::cout << "\n";
						exit(1);
					}
					gt(l) += d[mu]*g(r, i);
				}
			}
			gt(l) = gt(l)/(nodes*samples);
		}
		// for(int i = 0; i < knots; i++){
		// 	std::cout << gt(i) << " ";
		// }
		// std::cout << "\n";
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

REGISTER_KERNEL_BUILDER(Name("BSplineGrad").Device(DEVICE_CPU), BSplineGradOp);






// // the gradients are simply passed as additional arguments as
// // they are available in the Python function for registering the gradient operation.
// REGISTER_OP("BSplineGrad")
// 	.Input("grad: float32")
// 	.Input("input: float32")
// 	.Output("splinegrad: float32");

// /// \brief Implementation of an inner product gradient operation.
// /// Note that this operation is used in Python to register the gradient as
// /// this is not possible in C*+ right now.
// /// \param context
// /// \author David Stutz
// class BSplineGradOp : public OpKernel {
// 	public:
// 	/// \brief Constructor.
// 	/// \param context
// 	explicit BSplineGradOp(OpKernelConstruction* context) : OpKernel(context){}

// 	/// \brief Compute the inner product gradients.
// 	/// \param context
// 	void Compute(OpKernelContext* context) override{
// 		// Check number of inputs is correct
// 		DCHECK_EQ(2, context->num_inputs());
// 		// Fetch inputs
// 		const Tensor& input_tensor = context->input(1);
// 		auto input = input_tensor.flat<float>();
// 		const Tensor& grad_tensor = context->input(0);
// 		auto grad = grad_tensor.flat<float>();
		
// 		// Create an output tensor
// 		Tensor* output_tensor = NULL;
// 		OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
// 		auto output_flat = output_tensor->flat<float>();
		
// 		const int N = input.size();
// 		for(int i = 0; i < N; i++){
// 			if(input(i) < 0){
// 				output_flat(i) = -grad(i);
// 			}else{
// 				output_flat(i) = grad(i);
// 			}
// 		}
// 	}
// };

// REGISTER_KERNEL_BUILDER(Name("BSplineGrad").Device(DEVICE_CPU), BSplineGradOp);