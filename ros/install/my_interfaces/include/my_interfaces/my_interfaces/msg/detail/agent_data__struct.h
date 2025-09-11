// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from my_interfaces:msg/AgentData.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__AGENT_DATA__STRUCT_H_
#define MY_INTERFACES__MSG__DETAIL__AGENT_DATA__STRUCT_H_

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
// Member 'states'
// Member 'inputs'
// Member 'in_range_agents_ids'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/AgentData in the package my_interfaces.
/**
  *  AgentData Message
  * This ROS message contains information about an agent's state and control data for ergodic control applications.
 */
typedef struct my_interfaces__msg__AgentData
{
  /// Standard ROS header with timestamp and frame_id
  std_msgs__msg__Header header;
  double simulation_time;
  /// Ratio of real time for ergodic calculation over expected
  double delta_t_ts;
  int8_t num_of_states;
  int8_t num_of_inputs;
  rosidl_runtime_c__double__Sequence states;
  rosidl_runtime_c__double__Sequence inputs;
  double ergodic_cost;
  bool active_cbf_flag;
  rosidl_runtime_c__int8__Sequence in_range_agents_ids;
} my_interfaces__msg__AgentData;

// Struct for a sequence of my_interfaces__msg__AgentData.
typedef struct my_interfaces__msg__AgentData__Sequence
{
  my_interfaces__msg__AgentData * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} my_interfaces__msg__AgentData__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MY_INTERFACES__MSG__DETAIL__AGENT_DATA__STRUCT_H_
