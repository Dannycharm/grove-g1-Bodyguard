#pragma once
// Test-only API double. This is NOT MuJoCo and does not validate physics/sensors.
struct mjModel {
  int ngeom;
  int* body_mocapid;
  int* geom_bodyid;
  int* geom_contype;
  int* geom_conaffinity;
  int* geom_group;
};
struct mjData {
  double time;
  double* xpos;
  double* mocap_pos;
  double* mocap_quat;
};
constexpr int mjOBJ_BODY = 1;
int mj_name2id(const mjModel*, int, const char*);
void mj_kinematics(const mjModel*, mjData*);
