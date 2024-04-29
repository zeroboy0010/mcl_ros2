#include <rclcpp/rclcpp.hpp>
#include "robot_mcl_cpp/mcl.hpp"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("mcl");

    auto mcl = std::make_shared<mcl_ros::MCL>(node);
    double localizationHz = mcl->getLocalizationHz();
    rclcpp::Rate loop_rate(localizationHz);

    while (rclcpp::ok()) {
        rclcpp::spin_some(node);
        mcl->updateParticlesByMotionModel();
        mcl->setCanUpdateScan(false);
        mcl->calculateLikelihoodsByMeasurementModel();
        mcl->calculateEffectiveSampleSize();
        mcl->estimatePose();
        mcl->resampleParticles();
        mcl->estimateLocalizationCorrectness();
        mcl->publishROSMessages();
        mcl->broadcastTF();
        mcl->setCanUpdateScan(true);
        mcl->printResult();
        loop_rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
