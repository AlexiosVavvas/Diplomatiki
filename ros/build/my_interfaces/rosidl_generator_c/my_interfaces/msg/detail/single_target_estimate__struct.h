// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from my_interfaces:msg/SingleTargetEstimate.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__SINGLE_TARGET_ESTIMATE__STRUCT_H_
#define MY_INTERFACES__MSG__DETAIL__SINGLE_TARGET_ESTIMATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'position'
#include "geometry_msgs/msg/detail/point__struct.h"

/// Struct defined in msg/SingleTargetEstimate in the package my_interfaces.
/**
  * Target estimate message for ergodic exploration
  * Contains 3D position estimate and 3x3 covariance matrix
 */
typedef struct my_interfaces__msg__SingleTargetEstimate
{
  /// Target identification
  /// Unique identifier for the target
  int32_t target_id;
  /// Position estimate (3D)
  /// Estimated position [x, y, z]
  geometry_msgs__msg__Point position;
  /// Covariance matrix (3x3 flattened to 9 elements)
  /// Represents uncertainty in position estimate
  /// Order: [xx, xy, xz, yx, yy, yz, zx, zy, zz]
  /// 3x3 covariance matrix (row-major order)
  double covariance[9];
} my_interfaces__msg__SingleTargetEstimate;

// Struct for a sequence of my_interfaces__msg__SingleTargetEstimate.
typedef struct my_interfaces__msg__SingleTargetEstimate__Sequence
{
  my_interfaces__msg__SingleTargetEstimate * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} my_interfaces__msg__SingleTargetEstimate__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MY_INTERFACES__MSG__DETAIL__SINGLE_TARGET_ESTIMATE__STRUCT_H_
