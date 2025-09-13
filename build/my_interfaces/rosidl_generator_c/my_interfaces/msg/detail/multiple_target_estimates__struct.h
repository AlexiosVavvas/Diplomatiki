// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from my_interfaces:msg/MultipleTargetEstimates.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__MULTIPLE_TARGET_ESTIMATES__STRUCT_H_
#define MY_INTERFACES__MSG__DETAIL__MULTIPLE_TARGET_ESTIMATES__STRUCT_H_

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
// Member 'target_estimates'
// Member 'ground_truths'
#include "my_interfaces/msg/detail/single_target_estimate__struct.h"

/// Struct defined in msg/MultipleTargetEstimates in the package my_interfaces.
/**
  * MultipleTargetEstimates Message
  * This message contains an array of target estimates for multiple detected targets.
 */
typedef struct my_interfaces__msg__MultipleTargetEstimates
{
  /// Standard ROS header with timestamp and frame_id
  std_msgs__msg__Header header;
  int32_t num_of_targets;
  my_interfaces__msg__SingleTargetEstimate__Sequence target_estimates;
  my_interfaces__msg__SingleTargetEstimate__Sequence ground_truths;
} my_interfaces__msg__MultipleTargetEstimates;

// Struct for a sequence of my_interfaces__msg__MultipleTargetEstimates.
typedef struct my_interfaces__msg__MultipleTargetEstimates__Sequence
{
  my_interfaces__msg__MultipleTargetEstimates * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} my_interfaces__msg__MultipleTargetEstimates__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MY_INTERFACES__MSG__DETAIL__MULTIPLE_TARGET_ESTIMATES__STRUCT_H_
