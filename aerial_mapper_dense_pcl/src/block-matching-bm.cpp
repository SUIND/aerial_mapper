/*
 *    Filename: block-matching-bm.cpp
 *  Created on: Nov 01, 2017
 *      Author: Timo Hinzmann
 *   Institute: ETH Zurich, Autonomous Systems Lab
 */

// HEADER
#include "aerial-mapper-dense-pcl/block-matching-bm.h"

namespace stereo {
void BlockMatchingBM::computeDisparityMap(
    const RectifiedStereoPair& rectified_stereo_pair,
    DensifiedStereoPair* densified_stereo_pair) const {
  //LOG(INFO)<<"block-matching-bm.cpp 1";
  CHECK(densified_stereo_pair);
  //LOG(INFO)<<"block-matching-bm.cpp 1.1";
//   cv::cvtColor(rectified_stereo_pair.image_right, rectified_stereo_pair.image_right, cv::COLOR_BGR2GRAY);
//   cv::cvtColor(rectified_stereo_pair.image_left, rectified_stereo_pair.image_left, cv::COLOR_BGR2GRAY);
  //LOG(INFO)<<rectified_stereo_pair.image_left.type()<<" "<<rectified_stereo_pair.image_right.type();
  // Compute the disparity map.
  bm_->compute(rectified_stereo_pair.image_left,
               rectified_stereo_pair.image_right,
               densified_stereo_pair->disparity_map);
  //LOG(INFO)<<"block-matching-bm.cpp 1.2";
  densified_stereo_pair->disparity_map.convertTo(
      densified_stereo_pair->disparity_map, CV_32F);
  //LOG(INFO)<<"block-matching-bm.cpp 1.3";
  densified_stereo_pair->disparity_map =
      densified_stereo_pair->disparity_map / 16.0;
  //LOG(INFO)<<"block-matching-bm.cpp 2";
  // Create masked disparity map.
  cv::Mat disparity_map_masked(densified_stereo_pair->disparity_map.rows,
                               densified_stereo_pair->disparity_map.cols,
                               densified_stereo_pair->disparity_map.type());
  //LOG(INFO)<<"block-matching-bm.cpp 3";
  // Apply the rectification mask.
  disparity_map_masked.setTo(kMaxInvalidDisparity);
  densified_stereo_pair->disparity_map.copyTo(disparity_map_masked,
                                              rectified_stereo_pair.mask);
  densified_stereo_pair->disparity_map = disparity_map_masked;
}

}  // namespace stereo
