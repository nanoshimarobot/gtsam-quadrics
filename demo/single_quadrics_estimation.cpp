#include <gtsam_quadrics/geometry/ConstrainedDualQuadric.h>
#include <gtsam_quadrics/geometry/AlignedBox2.h>
#include <gtsam_quadrics/geometry/QuadricCamera.h>
#include <gtsam_quadrics/geometry/BoundingBoxFactor.h>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/CalibratedCamera.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <matplotlibcpp17/pyplot.h>
#include <matplotlibcpp17/mplot3d.h>
#include <matplotlibcpp17/animation.h>

#include <random>
#include <string>
#include <iostream>
#include <vector>

py::list draw_pose3(matplotlibcpp17::axes::Axes& ax, gtsam::Pose3& pose, float axis_length = 0.2) {
  auto rot3 = pose.rotation();
  auto t = pose.translation();
  std::vector<double> origin = {t.x(), t.y(), t.z()};

  // 各軸の終点を計算
  std::vector<double> x_axis = {rot3.matrix()(0, 0) * axis_length, rot3.matrix()(1, 0) * axis_length, rot3.matrix()(2, 0) * axis_length};
  std::vector<double> y_axis = {rot3.matrix()(0, 1) * axis_length, rot3.matrix()(1, 1) * axis_length, rot3.matrix()(2, 1) * axis_length};
  std::vector<double> z_axis = {rot3.matrix()(0, 2) * axis_length, rot3.matrix()(1, 2) * axis_length, rot3.matrix()(2, 2) * axis_length};

  py::list ret;
  ret.append(ax.quiver(Args(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2]), Kwargs("linewidth"_a = 1.0, "color"_a = "r")).unwrap());
  ret.append(ax.quiver(Args(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2]), Kwargs("linewidth"_a = 1.0, "color"_a = "g")).unwrap());
  ret.append(ax.quiver(Args(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2]), Kwargs("linewidth"_a = 1.0, "color"_a = "b")).unwrap());
  return ret;
}

py::list draw_ellipse(matplotlibcpp17::axes::Axes& ax, gtsam_quadrics::ConstrainedDualQuadric& dual_quadrics, std::string quad_color = "r") {
  py::list ret;
  double rx = dual_quadrics.radii().x();
  double ry = dual_quadrics.radii().y();
  double rz = dual_quadrics.radii().z();
  int num_points = 15;

  // パラメータ u, v によって楕円体の表面を表現する (0 <= u < 2π, 0 <= v <= π)
  std::vector<std::vector<double>> x, y, z;
  for (int i = 0; i < num_points; ++i) {
    std::vector<double> x_row, y_row, z_row;
    double u = 2 * M_PI * i / num_points;
    for (int j = 0; j < num_points; ++j) {
      double v = M_PI * j / num_points;

      // 楕円体のパラメータによる座標計算
      double x0 = rx * std::cos(u) * std::sin(v);
      double y0 = ry * std::sin(u) * std::sin(v);
      double z0 = rz * std::cos(v);

      // 回転と平行移動を適用して最終座標を取得
      gtsam::Vector3 point = dual_quadrics.pose().rotation().matrix() * gtsam::Vector3(x0, y0, z0) +
                             gtsam::Vector3(dual_quadrics.pose().translation().x(), dual_quadrics.pose().translation().y(), dual_quadrics.pose().translation().z());

      // 結果を格納
      x_row.push_back(point.x());
      y_row.push_back(point.y());
      z_row.push_back(point.z());
    }
    x.push_back(x_row);
    y.push_back(y_row);
    z.push_back(z_row);
  }

  auto _x = py::array(py::cast(std::move(x)));
  auto _y = py::array(py::cast(std::move(y)));
  auto _z = py::array(py::cast(std::move(z)));
  ret.append(ax.plot_wireframe(Args(_x, _y, _z), Kwargs("alpha"_a = 0.5, "color"_a = quad_color)).unwrap());
  return ret;
}

gtsam::Values optimize_graph(gtsam::NonlinearFactorGraph& graph, gtsam::Values& initial_estimate) {
  gtsam::LevenbergMarquardtParams lm_params;
  lm_params.setVerbosityLM("SUMMARY");
  gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial_estimate);
  return optimizer.optimize();
}

void marginalize(gtsam::NonlinearFactorGraph& graph, gtsam::Values& result) {
  gtsam::Marginals m(graph, result);
}

py::list draw_estimations(matplotlibcpp17::axes::Axes& ax, gtsam::Values& result) {
  py::list ret;
  for (const gtsam::Symbol& key : result.keys()) {
    if (key.chr() == 'x') {
      auto estimate_pose = result.at<gtsam::Pose3>(key);
      ret += draw_pose3(ax, estimate_pose);
    } else if (key.chr() == 'q') {
      auto estimate_quadrics = result.at<gtsam_quadrics::ConstrainedDualQuadric>(key);
      ret += draw_ellipse(ax, estimate_quadrics);
    }
  }
  return ret;
}

int main(void) {
  using namespace gtsam;
  using namespace gtsam_quadrics;
  using namespace matplotlibcpp17;
  using gtsam::symbol_shorthand::Q;
  using gtsam::symbol_shorthand::X;
  using matplotlibcpp17::animation::ArtistAnimation;

  py::scoped_interpreter guard{};
  auto plt = matplotlibcpp17::pyplot::import();

  matplotlibcpp17::mplot3d::import();
  auto fig = plt.figure();
  auto ax = fig.add_subplot(Args(), Kwargs("projection"_a = "3d"));

  auto odom_noise = noiseModel::Diagonal::Sigmas(Vector6(0.01, 0.1, 0.01, 0.2, 0.01, 0.2));
  auto bbox_noise = noiseModel::Diagonal::Sigmas(Vector4::Ones() * 3.0);
  auto quadrics_prior_noise = noiseModel::Diagonal::Sigmas(Vector9(0.2, 0.2, 0.2, 10.0, 10.0, 10.0, 0.3, 0.3, 0.3));
  auto robot_prior_noise = noiseModel::Diagonal::Sigmas(Vector6::Ones() * 1.0);

  boost::shared_ptr<Cal3_S2> calibration(new Cal3_S2(615.9603271484375, 616.227294921875, 0.0, 419.83026123046875, 245.1431427001953));

  // define ideal quadrics
  auto ideal_quadrics = ConstrainedDualQuadric(Pose3(Rot3(), Point3(3.0, -2.0, 0.5)), Point3(0.2, 0.2, 0.5));

  // define ideal robot halfway points
  std::vector<Pose3> halfway_poses{
    CalibratedCamera::LookatPose(Point3(0.0, -2.0, 0.5), Point3(3.0, -2.0, 0.5), Point3(0, 0, 1)),
    CalibratedCamera::LookatPose(Point3(0.0, -2.0, 0.5), Point3(3.0, -2.0, 0.5), Point3(0, 0, 1)),
    CalibratedCamera::LookatPose(Point3(3.0, 1.0, 0.5), Point3(3.0, -2.0, 0.5), Point3(0, 0, 1)),
    CalibratedCamera::LookatPose(Point3(6.0, -2.0, 0.5), Point3(3.0, -2.0, 0.5), Point3(0, 0, 1)),
    CalibratedCamera::LookatPose(Point3(3.0, -5.0, 0.5), Point3(3.0, -2.0, 0.5), Point3(0, 0, 1)),
    CalibratedCamera::LookatPose(Point3(0.0, -2.0, 0.5), Point3(3.0, -2.0, 0.5), Point3(0, 0, 1))};

  int between_n = 10;
  int deka_between_n = 50;
  std::vector<Pose3> ideal_trajectory;
  for (size_t i = 0; i < halfway_poses.size() - 1; ++i) {
    ideal_trajectory.push_back(halfway_poses[i]);
    for (size_t j = 0; j < ((i == 0) ? deka_between_n : between_n); ++j) {
      double perc = (j + 1) / double(between_n + 1);
      Pose3 new_pose = interpolate<Pose3>(halfway_poses[i], halfway_poses[i + 1], perc);
      ideal_trajectory.push_back(new_pose);
    }
  }
  ideal_trajectory.push_back(halfway_poses.back());

  std::vector<Pose3> ideal_odometry;
  for (size_t i = 0; i < ideal_trajectory.size() - 1; ++i) {
    ideal_odometry.push_back(ideal_trajectory[i].between(ideal_trajectory[i + 1]));
  }

  std::vector<AlignedBox2> ideal_bbox_list;
  for (auto pose : ideal_trajectory) {
    DualConic conic = QuadricCamera::project(ideal_quadrics, pose, calibration);
    AlignedBox2 bounds = conic.bounds();
    ideal_bbox_list.push_back(bounds);
  }

  // factor graph definition
  NonlinearFactorGraph graph;
  Values initial_estimate;

  // main process
  py::list artist_animation;

  graph.emplace_shared<PriorFactor<Pose3>>(X(0), ideal_trajectory[0], robot_prior_noise);
  initial_estimate.insert(X(0), ideal_trajectory[0]);

  auto res = optimize_graph(graph, initial_estimate);
  auto an = draw_estimations(ax, res);

  for (size_t i = 0; i < ideal_trajectory.size() - 1; ++i) {
    std::cout << "proc " << i << std::endl;
    // odom
    graph.emplace_shared<BetweenFactor<Pose3>>(X(i), X(i + 1), ideal_odometry[i], odom_noise);
    initial_estimate.insert(X(i + 1), ideal_trajectory[i + 1]);

    graph.emplace_shared<BoundingBoxFactor>(ideal_bbox_list[i + 1], calibration, X(i + 1), Q(0), bbox_noise);
    if (!initial_estimate.exists(Q(0))) {
      graph.emplace_shared<PriorFactor<ConstrainedDualQuadric>>(Q(0), ideal_quadrics, quadrics_prior_noise);
    }
    initial_estimate.insert_or_assign(Q(0), ideal_quadrics);  // onlineなら前回の推定値をinitialに入れたほうがいい?

    auto res = optimize_graph(graph, initial_estimate);
    auto an = draw_estimations(ax, res);
    artist_animation.append(an + draw_ellipse(ax, ideal_quadrics, "b"));
  }
  auto ani = ArtistAnimation(Args(fig.unwrap(), artist_animation), Kwargs("interval"_a = 300));
  ax.set_aspect(Args("equal"));
  plt.show();
  return 1;
}