// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from my_interfaces:msg/MultipleObstacles.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__MULTIPLE_OBSTACLES__STRUCT_H_
#define MY_INTERFACES__MSG__DETAIL__MULTIPLE_OBSTACLES__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'obstacles'
#include "my_interfaces/msg/detail/single_obstacle__struct.h"

/// Struct defined in msg/MultipleObstacles in the package my_interfaces.
/**
  * Multiple obstacles message for ergodic exploration
  * Contains an array of SingleObstacle messages
 */
typedef struct my_interfaces__msg__MultipleObstacles
{
  /// Number of obstacles in the array
  int32_t num_of_obstacles;
  /// Array of obstacles
  my_interfaces__msg__SingleObstacle__Sequence obstacles;
} my_interfaces__msg__MultipleObstacles;

// Struct for a sequence of my_interfaces__msg__MultipleObstacles.
typedef struct my_interfaces__msg__MultipleObstacles__Sequence
{
  my_interfaces__msg__MultipleObstacles * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} my_interfaces__msg__MultipleObstacles__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MY_INTERFACES__MSG__DETAIL__MULTIPLE_OBSTACLES__STRUCT_H_
