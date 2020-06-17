#include "image_enhance/guided_filter.h"

static cv::Mat boxfilter(const cv::Mat &I, int r)
{
	cv::Mat result;
	cv::blur(I, result, cv::Size(r, r));
	return result;
}

static cv::Mat convertTo(const cv::Mat &mat, int depth)
{
	if (mat.depth() == depth)
		return mat;

	cv::Mat result;
	mat.convertTo(result, depth);
	return result;
}

class GuidedFilterImpl
{
public:
	virtual ~GuidedFilterImpl() {}

	cv::Mat filter(const cv::Mat &p, int depth);

protected:
	int Idepth;
	cv::Size original_image_size;

private:
	virtual cv::Mat filterSingleChannel(const cv::Mat &p) const = 0;
};

class GuidedFilterMono : public GuidedFilterImpl
{
public:
	GuidedFilterMono(const cv::Mat &I, int r, double eps);

private:
	virtual cv::Mat filterSingleChannel(const cv::Mat &p) const;

private:
	int r;
	double eps;
	cv::Mat I, mean_I, var_I;
};

class GuidedFilterColor : public GuidedFilterImpl
{
public:
	GuidedFilterColor(const cv::Mat &I, int r, double eps, int resize_factor);

private:
	virtual cv::Mat filterSingleChannel(const cv::Mat &p) const;

private:
	std::vector<cv::Mat> Ichannels;
	int r;
	int sub_r;
	cv::Size resized_size;
	double eps;
	cv::Mat mean_I_r, mean_I_g, mean_I_b;
	cv::Mat invrr, invrg, invrb, invgg, invgb, invbb;
};


cv::Mat GuidedFilterImpl::filter(const cv::Mat &p, int depth)
{
	std::cout << "Filter Start" << std::endl;
	cv::Mat p2 = convertTo(p, Idepth);

	cv::Mat result;
	if (p.channels() == 1)
	{
		result = filterSingleChannel(p2);
	}
	else
	{
		std::vector<cv::Mat> pc;
		cv::split(p2, pc);

		for (std::size_t i = 0; i < pc.size(); ++i)
			pc[i] = filterSingleChannel(pc[i]);

		std::cout << "Filter Merge" << std::endl;
		cv::merge(pc, result);
	}

	std::cout << "Resizing Back to Original Size" << std::endl;
	cv::resize(result, result, original_image_size, 0,0, CV_INTER_NN);

	return convertTo(result, depth == -1 ? p.depth() : depth);
}

GuidedFilterMono::GuidedFilterMono(const cv::Mat &origI, int r, double eps) : r(r), eps(eps)
{
	original_image_size = origI.size();
	if (origI.depth() == CV_32F || origI.depth() == CV_64F)
		I = origI.clone();
	else
		I = convertTo(origI, CV_32F);

	Idepth = I.depth();

	mean_I = boxfilter(I, r);
	cv::Mat mean_II = boxfilter(I.mul(I), r);
	var_I = mean_II - mean_I.mul(mean_I);
}

cv::Mat GuidedFilterMono::filterSingleChannel(const cv::Mat &p) const
{
	cv::Mat mean_p = boxfilter(p, r);
	cv::Mat mean_Ip = boxfilter(I.mul(p), r);
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p); // this is the covariance of (I, p) in each local patch.

	cv::Mat a = cov_Ip / (var_I + eps); // Eqn. (5) in the paper;
	cv::Mat b = mean_p - a.mul(mean_I); // Eqn. (6) in the paper;

	cv::Mat mean_a = boxfilter(a, r);
	cv::Mat mean_b = boxfilter(b, r);

	return mean_a.mul(I) + mean_b;
}

GuidedFilterColor::GuidedFilterColor(const cv::Mat &origI, int r, double eps, int resize_factor) : r(r), eps(eps)
{
	original_image_size = origI.size();

	cv::Mat reformated_I(origI.cols, origI.rows, CV_64FC1);

	if (origI.depth() != CV_32F && origI.depth() != CV_64F)
		origI.convertTo(reformated_I, CV_32F);
	else
		reformated_I = origI.clone();
	
	// std::cout << "Resizing original image" << std::endl;
	resized_size = cv::Size(origI.cols/resize_factor, origI.rows/resize_factor);
	cv::Mat resized_I(origI.cols/resize_factor, origI.rows/resize_factor, CV_64FC1);
	cv::resize(reformated_I, resized_I, resized_size, 0, 0, CV_INTER_NN);

	Idepth = resized_I.depth();

	sub_r = r/resize_factor;

	cv::split(resized_I, Ichannels);

	mean_I_r = boxfilter(Ichannels[0], sub_r);
	mean_I_g = boxfilter(Ichannels[1], sub_r);
	mean_I_b = boxfilter(Ichannels[2], sub_r);

	// variance of I in each local patch: the matrix Sigma in Eqn (14).
	// Note the variance in each local patch is a 3x3 symmetric matrix:
	//           rr, rg, rb
	//   Sigma = rg, gg, gb
	//           rb, gb, bb
	cv::Mat var_I_rr = boxfilter(Ichannels[0].mul(Ichannels[0]), sub_r) - mean_I_r.mul(mean_I_r) + eps;
	cv::Mat var_I_rg = boxfilter(Ichannels[0].mul(Ichannels[1]), sub_r) - mean_I_r.mul(mean_I_g);
	cv::Mat var_I_rb = boxfilter(Ichannels[0].mul(Ichannels[2]), sub_r) - mean_I_r.mul(mean_I_b);
	cv::Mat var_I_gg = boxfilter(Ichannels[1].mul(Ichannels[1]), sub_r) - mean_I_g.mul(mean_I_g) + eps;
	cv::Mat var_I_gb = boxfilter(Ichannels[1].mul(Ichannels[2]), sub_r) - mean_I_g.mul(mean_I_b);
	cv::Mat var_I_bb = boxfilter(Ichannels[2].mul(Ichannels[2]), sub_r) - mean_I_b.mul(mean_I_b) + eps;

	// Inverse of Sigma + eps * I
	invrr = var_I_gg.mul(var_I_bb) - var_I_gb.mul(var_I_gb);
	invrg = var_I_gb.mul(var_I_rb) - var_I_rg.mul(var_I_bb);
	invrb = var_I_rg.mul(var_I_gb) - var_I_gg.mul(var_I_rb);
	invgg = var_I_rr.mul(var_I_bb) - var_I_rb.mul(var_I_rb);
	invgb = var_I_rb.mul(var_I_rg) - var_I_rr.mul(var_I_gb);
	invbb = var_I_rr.mul(var_I_gg) - var_I_rg.mul(var_I_rg);

	cv::Mat covDet = invrr.mul(var_I_rr) + invrg.mul(var_I_rg) + invrb.mul(var_I_rb);

	invrr /= covDet;
	invrg /= covDet;
	invrb /= covDet;
	invgg /= covDet;
	invgb /= covDet;
	invbb /= covDet;
}

cv::Mat GuidedFilterColor::filterSingleChannel(const cv::Mat &p) const
{
	cv::Mat resized_p;
	cv::resize(p, resized_p, resized_size, 0, 0, CV_INTER_NN);
	// std::cout << "Filter Single Channel" << std::endl;
	cv::Mat mean_p = boxfilter(resized_p, sub_r);

	// std::cout << "Calc Mean" << std::endl;
	cv::Mat mean_Ip_r = boxfilter(Ichannels[0].mul(resized_p), sub_r);
	cv::Mat mean_Ip_g = boxfilter(Ichannels[1].mul(resized_p), sub_r);
	cv::Mat mean_Ip_b = boxfilter(Ichannels[2].mul(resized_p), sub_r);

	// std::cout << "Calc Cov" << std::endl;
	// covariance of (I, p) in each local patch.
	cv::Mat cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);
	cv::Mat cov_Ip_g = mean_Ip_g - mean_I_g.mul(mean_p);
	cv::Mat cov_Ip_b = mean_Ip_b - mean_I_b.mul(mean_p);

	// std::cout << "Calc a" << std::endl;
	cv::Mat a_r = invrr.mul(cov_Ip_r) + invrg.mul(cov_Ip_g) + invrb.mul(cov_Ip_b);
	cv::Mat a_g = invrg.mul(cov_Ip_r) + invgg.mul(cov_Ip_g) + invgb.mul(cov_Ip_b);
	cv::Mat a_b = invrb.mul(cov_Ip_r) + invgb.mul(cov_Ip_g) + invbb.mul(cov_Ip_b);

	// std::cout << "Calc b" << std::endl;
	cv::Mat b = mean_p - a_r.mul(mean_I_r) - a_g.mul(mean_I_g) - a_b.mul(mean_I_b); // Eqn. (15) in the paper;

	// std::cout << "Assembling" << std::endl;
	return (boxfilter(a_r, sub_r).mul(Ichannels[0])
		+ boxfilter(a_g, sub_r).mul(Ichannels[1])
		+ boxfilter(a_b, sub_r).mul(Ichannels[2])
		+ boxfilter(b, sub_r));  // Eqn. (16) in the paper;
}


GuidedFilter::GuidedFilter(const cv::Mat &I, int r, double eps, int resize_factor)
{
	CV_Assert(I.channels() == 1 || I.channels() == 3);

	if (I.channels() == 1)
		impl_ = new GuidedFilterMono(I, 2 * r + 1, eps);
	else
		impl_ = new GuidedFilterColor(I, 2 * r + 1, eps, resize_factor);
}

GuidedFilter::~GuidedFilter()
{
	delete impl_;
}

cv::Mat GuidedFilter::filter(const cv::Mat &p, int depth) const
{
	return impl_->filter(p, depth);
}

cv::Mat guidedFilter(const cv::Mat &I, const cv::Mat &p, int r, double eps, int resize_factor, int depth)
{
	return GuidedFilter(I, r, eps, resize_factor).filter(p, depth);
}