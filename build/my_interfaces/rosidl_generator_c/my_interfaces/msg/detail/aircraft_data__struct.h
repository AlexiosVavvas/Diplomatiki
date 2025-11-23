// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from my_interfaces:msg/AircraftData.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__AIRCRAFT_DATA__STRUCT_H_
#define MY_INTERFACES__MSG__DETAIL__AIRCRAFT_DATA__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"

/// Struct defined in msg/AircraftData in the package my_interfaces.
/**
  * AircraftData Message
  * Complete flight state information for fixed-wing aircraft in NED frame
 */
typedef struct my_interfaces__msg__AircraftData
{
  /// Standard ROS header with timestamp and frame_id
  std_msgs__msg__Header header;
  /// Position (NED frame, meters)
  /// Position north (m)
  double north;
  /// Position east (m)
  double east;
  /// Position down (m)
  double down;
  /// Altitude above reference (m)
  double altitude;
  /// Attitude (radians and degrees)
  /// Roll angle (rad)
  double roll;
  /// Pitch angle (rad)
  double pitch;
  /// Yaw angle (rad)
  double yaw;
  /// Roll angle (deg)
  double roll_deg;
  /// Pitch angle (deg)
  double pitch_deg;
  /// Yaw angle (deg)
  double yaw_deg;
  /// Body frame velocities (m/s)
  /// Velocity along aircraft nose (m/s)
  double u_forward;
  /// Velocity to the right (m/s)
  double v_sideways;
  /// Velocity through aircraft belly (m/s)
  double w_downward;
  /// Total airspeed (m/s)
  double airspeed;
  /// NED frame velocities (m/s)
  /// Velocity north (m/s)
  double v_north;
  /// Velocity east (m/s)
  double v_east;
  /// Velocity down (m/s)
  double v_down;
  /// Climb rate, positive up (m/s)
  double climb_rate;
  /// Horizontal ground speed (m/s)
  double ground_speed;
  /// Angular rates (rad/s and deg/s)
  /// Roll rate (rad/s)
  double p_roll_rate;
  /// Pitch rate (rad/s)
  double q_pitch_rate;
  /// Yaw rate (rad/s)
  double r_yaw_rate;
  /// Roll rate (deg/s)
  double p_deg_s;
  /// Pitch rate (deg/s)
  double q_deg_s;
  /// Yaw rate (deg/s)
  double r_deg_s;
  /// Aerodynamic angles
  /// Angle of attack (rad)
  double alpha;
  /// Sideslip angle (rad)
  double beta;
  /// Angle of attack (deg)
  double alpha_deg;
  /// Sideslip angle (deg)
  double beta_deg;
} my_interfaces__msg__AircraftData;

// Struct for a sequence of my_interfaces__msg__AircraftData.
typedef struct my_interfaces__msg__AircraftData__Sequence
{
  my_interfaces__msg__AircraftData * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} my_interfaces__msg__AircraftData__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MY_INTERFACES__MSG__DETAIL__AIRCRAFT_DATA__STRUCT_H_
