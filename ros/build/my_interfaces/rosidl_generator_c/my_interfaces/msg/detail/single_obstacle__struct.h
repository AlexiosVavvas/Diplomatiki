// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from my_interfaces:msg/SingleObstacle.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__SINGLE_OBSTACLE__STRUCT_H_
#define MY_INTERFACES__MSG__DETAIL__SINGLE_OBSTACLE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'obs_type'
// Member 'obs_name'
#include "rosidl_runtime_c/string.h"
// Member 'position'
#include "geometry_msgs/msg/detail/point__struct.h"
// Member 'dimensions'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/SingleObstacle in the package my_interfaces.
/**
  * Single obstacle message for ergodic exploration
  * Supports circle, rectangle, and wall obstacle types
 */
typedef struct my_interfaces__msg__SingleObstacle
{
  /// Type of obstacle: "circle", "rectangle", or "wall"
  rosidl_runtime_c__String obs_type;
  /// Name/ID of the obstacle (optional, for debugging)
  rosidl_runtime_c__String obs_name;
  /// Obstacle position [x, y, z]
  geometry_msgs__msg__Point position;
  /// Dimensions based on type:
  /// - circle:
  /// - rectangle: [width, height]
  /// - wall: [normal_x, normal_y]
  rosidl_runtime_c__double__Sequence dimensions;
  /// Potential field parameter
  double kappa;
  /// Obstacle vicinity distance
  double rho0;
} my_interfaces__msg__SingleObstacle;

// Struct for a sequence of my_interfaces__msg__SingleObstacle.
typedef struct my_interfaces__msg__SingleObstacle__Sequence
{
  my_interfaces__msg__SingleObstacle * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} my_interfaces__msg__SingleObstacle__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MY_INTERFACES__MSG__DETAIL__SINGLE_OBSTACLE__STRUCT_H_
