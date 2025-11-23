// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from my_interfaces:msg/ObsAvoidanceDebug.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__OBS_AVOIDANCE_DEBUG__STRUCT_H_
#define MY_INTERFACES__MSG__DETAIL__OBS_AVOIDANCE_DEBUG__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'beta'
// Member 'u_safe'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/ObsAvoidanceDebug in the package my_interfaces.
/**
  * ObsAvoidanceDebug Message
  * This ROS message contains information about an agent's state regarding the obstacle avoidance tuning
 */
typedef struct my_interfaces__msg__ObsAvoidanceDebug
{
  double psi;
  double hddot;
  double two_alpha_h_hdot;
  double alpha2_h;
  rosidl_runtime_c__double__Sequence beta;
  rosidl_runtime_c__double__Sequence u_safe;
} my_interfaces__msg__ObsAvoidanceDebug;

// Struct for a sequence of my_interfaces__msg__ObsAvoidanceDebug.
typedef struct my_interfaces__msg__ObsAvoidanceDebug__Sequence
{
  my_interfaces__msg__ObsAvoidanceDebug * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} my_interfaces__msg__ObsAvoidanceDebug__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MY_INTERFACES__MSG__DETAIL__OBS_AVOIDANCE_DEBUG__STRUCT_H_
