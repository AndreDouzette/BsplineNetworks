#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("BSpline")
	.Input("x: float")
	.Input("t: float")
	.Input("c: float")
	.Input("p: int32")
	.Output("s: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
		c->set_output(0, c->input(0));
		return Status::OK();
	});

class BSplineOp: public OpKernel{
	public:
	explicit BSplineOp(OpKernelConstruction* context): OpKernel(context){}
	void Compute(OpKernelContext* context) override{
		// Grab the input tensor
		const Tensor& X = context->input(0);
		const Tensor& T = context->input(1);
		const Tensor& C = context->input(2);
		const Tensor& P = context->input(3);
		auto x = X.matrix<float>();
		auto t = T.vec<float>();
		auto c = C.matrix<float>();
		auto p = P.scalar<int>()(0);
		
		// Create an output tensor
		Tensor* S = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, X.shape(), &S));
		auto s = S->matrix<float>();
		
		const int internalKnots = C.shape().dim_size(1);
		const int knots = t.size();
		const int samples = S->shape().dim_size(0);
		const int nodes = S->shape().dim_size(1);
		
		// std::cout << S->shape().dim_size(0) << " " << S->shape().dim_size(1) << "\n";
		
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
				if(xx < t(p)){
					s(r, i) = c(i, 0);
					continue;
				}
				if(xx >= t(knots - p - 1)){
					s(r, i) = c(i, internalKnots - 1);
					continue;
				}
				mu = binarySearch(xx, t);
				// if(t(mu + 1) - t(mu) <= 1e-5){
				// 	std::cout << xx << " in " << mu << "[" << t(mu) << ", " << t(mu + 1) << ") 2\n";
				// }
				for(int j = mu - p; j <= mu; j++){
					d[j] = c(i, j);
				}
				for(int k = p; k > 0; k--){
					for(int j = mu; j > mu - k; j--){
						a = (xx - t(j))/(t(j + k) - t(j));
						d[j] = (1 - a)*d[j - 1] + a*d[j];
					}
				}
				s(r, i) =  d[mu];
			}
		}
		// for(int i = 0; i < knots; i++){
		// 	if(t(i) < 0){
		// 		std::cout << i << ":" << t(i) << " ";
		// 	}
		// }
		// for(int i = 0; i < knots; i++){
		// 	if(t(i) < 0){
		// 		std::cout << "\n";
		// 		break;
		// 	}
		// }
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

REGISTER_KERNEL_BUILDER(Name("BSpline").Device(DEVICE_CPU), BSplineOp);


// #include "tensorflow/core/framework/op_kernel.h"
// #include "tensorflow/core/framework/tensor_shape.h"
// #include "tensorflow/core/platform/default/logging.h"
// #include "tensorflow/core/framework/shape_inference.h"

// using namespace tensorflow;

// REGISTER_OP("BSpline")
// 	.Input("input: float")
// 	.Input("weights: float")
// 	.Output("inner_product: float")
// 	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
// 		shape_inference::ShapeHandle input_shape;
// 		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));

// 		shape_inference::ShapeHandle weight_shape;
// 		TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &weight_shape));

// 		shape_inference::DimensionHandle output_rows = c->Dim(weight_shape, 0);

// 		shape_inference::DimensionHandle input_rows = c->Dim(input_shape, 0);
// 		shape_inference::DimensionHandle weight_cols = c->Dim(weight_shape, 1);
// 		shape_inference::DimensionHandle merged;
// 		TF_RETURN_IF_ERROR(c->Merge(input_rows, weight_cols, &merged));

// 		c->set_output(0, c->Matrix(output_rows, 1));
// 		return Status::OK();
//   });

// /// \brief Implementation of an inner product operation.
// /// \param context
// /// \author David Stutz
// class BSplineOp: public OpKernel{
// public:
// 	/// \brief Constructor.
// 	/// \param context
// 	explicit BSplineOp(OpKernelConstruction* context):OpKernel(context){}
// 	/// \brief Compute the inner product.
// 	/// \param context
// 	void Compute(OpKernelContext* context) override{
// 		// some checks to be sure ...
// 		DCHECK_EQ(2, context->num_inputs());

// 		// get the input tensor
// 		const Tensor& input = context->input(0);

// 		// get the weight tensor
// 		const Tensor& weights = context->input(1);

// 		// check shapes of input and weights
// 		const TensorShape& input_shape = input.shape();
// 		const TensorShape& weights_shape = weights.shape();

// 		// check input is a standing vector
// 		DCHECK_EQ(input_shape.dims(), 2);
// 		DCHECK_EQ(input_shape.dim_size(1), 1);

// 		// check weights is matrix of correct size
// 		DCHECK_EQ(weights_shape.dims(), 2);
// 		DCHECK_EQ(input_shape.dim_size(0), weights_shape.dim_size(1));

// 		// create output shape
// 		TensorShape output_shape;
// 		output_shape.AddDim(weights_shape.dim_size(0));
// 		output_shape.AddDim(1);
		
// 		// create output tensor
// 		Tensor* output = NULL;
// 		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

// 		// get the corresponding Eigen tensors for data access
// 		auto input_tensor = input.matrix<float>();
// 		auto weights_tensor = weights.matrix<float>();
// 		auto output_tensor = output->matrix<float>();

// 		for(int i = 0; i < output->shape().dim_size(0); i++){
// 			output_tensor(i, 0) = 0;
// 			for(int j = 0; j < weights.shape().dim_size(1); j++){
// 				output_tensor(i, 0) += weights_tensor(i, j)*input_tensor(j, 0);
// 			}
// 		}
// 	}
// };

// REGISTER_KERNEL_BUILDER(Name("InnerProduct").Device(DEVICE_CPU), BSplineOp);