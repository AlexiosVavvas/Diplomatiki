// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from my_interfaces:msg/CkTable.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__CK_TABLE__STRUCT_H_
#define MY_INTERFACES__MSG__DETAIL__CK_TABLE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'ck_values'
// Member 'ck_values_average_in_range'
#include "rosidl_runtime_c/primitives_sequence.h"
// Member 'position'
#include "geometry_msgs/msg/detail/point__struct.h"

/// Struct defined in msg/CkTable in the package my_interfaces.
/**
  * Table of floats message
  * Represents a square table/matrix of float values
 */
typedef struct my_interfaces__msg__CkTable
{
  /// Size of the square table (size x size)
  int32_t table_size;
  /// Flattened array of float values (row-major order)
  rosidl_runtime_c__double__Sequence ck_values;
  /// Mean Ck array for agents in range as seen by a particular agent
  rosidl_runtime_c__double__Sequence ck_values_average_in_range;
  /// Total ergodic cost for all agents now
  double total_erg_cost;
  /// Total ergodic cost for all in range agents now
  double total_erg_cost_in_range;
  /// Reduction percentage of Total ergodic cost for all in range agents now. For example if init is 100 and now 90 -> 10%
  double erg_cost_reduction_perc;
  /// Current Agent position [x, y, z] (used to simulate short antenna range)
  geometry_msgs__msg__Point position;
} my_interfaces__msg__CkTable;

// Struct for a sequence of my_interfaces__msg__CkTable.
typedef struct my_interfaces__msg__CkTable__Sequence
{
  my_interfaces__msg__CkTable * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} my_interfaces__msg__CkTable__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MY_INTERFACES__MSG__DETAIL__CK_TABLE__STRUCT_H_
