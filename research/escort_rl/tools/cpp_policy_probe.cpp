// Offline only: test access exposes existing buffers without changing production sources.
#include <onnxruntime_cxx_api.h>
#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#define private public
#include "g1_controllers/agile_policy.hpp"
#undef private
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    g1_controllers::AgilePolicy policy(argv[1]);
    std::ifstream in(argv[2]);
    if (std::filesystem::exists(argv[3])) return 6;
    std::ofstream out(argv[3]);
    if (!in || !out) return 3;
    out << std::setprecision(9);
    int reset;
    while (in >> reset) {
        if (reset) policy.reset();
        g1_controllers::PolicyObservation o;
        for (auto* field : {&o.root_quat_wxyz}) for (auto& v : *field) in >> v;
        for (auto& v : o.root_ang_vel_b) in >> v;
        for (auto& v : o.velocity_command) in >> v;
        for (auto& v : o.joint_position) in >> v;
        for (auto& v : o.joint_velocity) in >> v;
        if (!in) return 4;
        g1_controllers::PolicyAction a;
        if (!policy.run(o,a)) return 5;
        for (auto& field : {a.joint_position,a.kp,a.kd}) for (auto v : field) out << v << ' ';
        for (auto v : policy.state_out_) out << v << ' ';
        out << '\n';
    }
}
