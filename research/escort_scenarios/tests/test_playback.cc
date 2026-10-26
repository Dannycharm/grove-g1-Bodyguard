// Offline unit tests of actual C++ playback using a tiny MuJoCo API double.
#include "pedestrian_playback.h"
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {
int calls=0;
void check(bool value) { if (!value) throw std::runtime_error("assertion failed"); }
template<class F> void rejects(F run) {
  bool failed=false; try { run(); } catch (const std::exception&) { failed=true; }
  check(failed);
}
}
int mj_name2id(const mjModel*, int, const char* name) {
  std::string n(name);
  if (n == "pelvis") return 1;
  if (n == "escort_client") return 2;
  if (n == "escort_pedestrian_1") return 3;
  if (n == "escort_pedestrian_2") return 4;
  return -1;
}
void mj_kinematics(const mjModel*, mjData*) { ++calls; }
int main(int argc, char** argv) {
  try {
    check(argc == 2);
    const std::string path=argv[1];
    int slots[]={-1,-1,0,1,2}, bodies[]={2,3,4}, zero[]={0,0,0}, group[]={2,2,2};
    mjModel model{3,slots,bodies,zero,zero,group};
    double xpos[15]={}, pos[9]={}, quat[12]={};
    mjData data{100,xpos,pos,quat};
    grove_g1::PedestrianPlayback playback(path);
    playback.Bind(&model,&data);
    check(playback.State() == "ready" && playback.Time() == 0);
    check(std::abs(pos[0]+1.4) < 1e-12 && std::abs(pos[1]-1.35) < 1e-12);
    data.time=101; playback.Advance(&model,&data); check(playback.Time() == 0);
    playback.Command("start",&model,&data);
    data.time=102; playback.Advance(&model,&data); check(playback.Time() == 1);
    double x=pos[0]; check(x > -1.4 && x < 1.4);
    playback.Command("pause",&model,&data);
    data.time=110; playback.Advance(&model,&data); check(playback.Time() == 1 && pos[0] == x);
    // Reset refuses the current G1 footprint, and rejection preserves playback time.
    xpos[3]=-1.4; xpos[4]=0.8;
    rejects([&]{playback.Command("reset",&model,&data);}); check(playback.Time() == 1);
    xpos[3]=xpos[4]=0;
    playback.Command("reset",&model,&data); check(playback.Time() == 0 && playback.State()=="ready");
    playback.Command("replay",&model,&data);
    data.time=111; playback.Advance(&model,&data); check(pos[0] == x);
    rejects([&]{playback.Command("replay",&model,&data);});
    rejects([&]{playback.Command("unknown",&model,&data);});
    for (int i=0;i<200;++i) {
      data.time+=0.05; playback.Advance(&model,&data);
      for (int a=0;a<3;++a) check(std::abs(pos[3*a]) <= 1.58 && std::abs(pos[3*a+1]) <= 1.58);
    }
    check(playback.State()=="finished");
    data.time=1; rejects([&]{playback.Advance(&model,&data);});
    // Missing mocap mapping and accidental collidability are rejected.
    slots[2]=-1;
    grove_g1::PedestrianPlayback missing(path); rejects([&]{missing.Bind(&model,&data);});
    slots[2]=0; zero[0]=1;
    grove_g1::PedestrianPlayback collidable(path); rejects([&]{collidable.Bind(&model,&data);});
    zero[0]=0;
    data.time=1000;
    grove_g1::PedestrianPlayback golden(path); golden.Bind(&model,&data);
    golden.Command("start",&model,&data);
    std::ifstream expected(path+".golden"); check(expected.good());
    double t,ex,ey,eqw,eqz;
    while (expected >> t >> ex >> ey >> eqw >> eqz) {
      data.time=1000+t; golden.Advance(&model,&data);
      check(std::abs(pos[0]-ex)<1e-10 && std::abs(pos[1]-ey)<1e-10);
      check(std::abs(quat[0]-eqw)<1e-10 && std::abs(quat[3]-eqz)<1e-10);
    }
    auto root=YAML::LoadFile(path); root["actors"][0]["body"]="escort_missing";
    const auto invalid=path+".invalid";
    { std::ofstream output(invalid); output << root; }
    rejects([&]{grove_g1::PedestrianPlayback bad(invalid);});
    // Exercise the real mailbox wrapper, including ignoring stale startup commands.
    const auto directory=std::filesystem::path(path).parent_path();
    const auto control=directory/"control.yaml";
    auto write_command=[&](int seq,const char* action) {
      auto stamp=std::filesystem::file_time_type::clock::now()+std::chrono::seconds(seq);
      { std::ofstream file(control.string()+".tmp"); file << "sequence: " << seq << "\naction: " << action << "\n"; }
      std::filesystem::rename(control.string()+".tmp",control);
      std::filesystem::last_write_time(control,stamp);
    };
    write_command(100,"replay");
    setenv("GROVE_G1_PEDESTRIAN_SCENARIO",path.c_str(),1);
    grove_g1::UpdatePedestrians(&model,&data,true);
    check(YAML::LoadFile((directory/"status.yaml").string())["state"].as<std::string>()=="ready");
    write_command(101,"start"); grove_g1::UpdatePedestrians(&model,&data,true);
    check(YAML::LoadFile((directory/"status.yaml").string())["state"].as<std::string>()=="playing");
    data.time+=1; grove_g1::UpdatePedestrians(&model,&data,false);
    write_command(102,"pause"); grove_g1::UpdatePedestrians(&model,&data,true);
    check(YAML::LoadFile((directory/"status.yaml").string())["scenario_time_s"].as<double>()==1);
    write_command(103,"reset"); grove_g1::UpdatePedestrians(&model,&data,true);
    check(YAML::LoadFile((directory/"status.yaml").string())["state"].as<std::string>()=="ready");
    check(data.time==1011);  // No G1/simulation clock reset by any scenario command.
    check(calls > 0);
    std::cout << "PASS: discovery, pose, pause/reset/replay, bounds, deterministic playback, invalid configurations (MuJoCo API double only)\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
