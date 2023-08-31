/*
 *    Filename: main-generate-pix4d-geofile.cc
 *  Created on: Jun 25, 2017
 *      Author: Timo Hinzmann
 *   Institute: ETH Zurich, Autonomous Systems Lab
 */

// SYSTEM
#include <memory>

// NON-SYSTEM
#include <aerial-mapper-io/aerial-mapper-io.h>
#include <aerial-mapper-ortho/ortho-backward-grid.h>
#include <gflags/gflags.h>
#include <ros/ros.h>

DEFINE_string(util_data_directory, "",
              "Directory to poses, images, and calibration file.");
DEFINE_string(util_filename_poses, "",
              "Name of the file that contains positions and orientations for "
              "every camera in the global/world frame, i.e. T_G_B");
DEFINE_string(util_prefix_images, "",
              "Prefix of the images to be loaded, e.g. 'images_'");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InstallFailureSignalHandler();

  ros::init(argc, argv, "util_generate_pix4d_geofile");
  ros::Time::init();


  //ROS_INFO("Starting the util_generate_pix4d_geofile node");
  // Parse input parameters.
  const std::string& base = "/home/axelwagner/Downloads/cadastre_gray/";
  const std::string& filename_poses = "opt_poses.txt";
  const std::string& filename_images = base + "image_";

  // Load body poses from file.
  Poses T_G_Bs;
  const std::string& path_filename_poses = base + filename_poses;
  io::PoseFormat pose_format = io::PoseFormat::Standard;
  io::AerialMapperIO io_handler;
  io_handler.loadPosesFromFile(pose_format, path_filename_poses, &T_G_Bs);

  // Load images from file.
  size_t num_poses = T_G_Bs.size();
  Images images;
  io_handler.loadImagesFromFile(filename_images, num_poses, &images);

  // Export poses.
  io_handler.exportPix4dGeofile(T_G_Bs, images);

  return 0;
}
