// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from my_interfaces:msg/JoystickData.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__JOYSTICK_DATA__STRUCT_H_
#define MY_INTERFACES__MSG__DETAIL__JOYSTICK_DATA__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/JoystickData in the package my_interfaces.
/**
  * Joystick control data
  * All stick values normalized to -1.0 to 1.0 range
  * Switch is binary 0 or 1
 */
typedef struct my_interfaces__msg__JoystickData
{
  /// Throttle stick (channel 1)
  double throttle;
  /// Aileron stick (channel 3)
  double aileron;
  /// Elevator stick (channel 5)
  double elevator;
  /// Rudder stick (channel 7)
  double rudder;
  /// Switch (channel 9) - binary 0/1
  int32_t switch_state;
} my_interfaces__msg__JoystickData;

// Struct for a sequence of my_interfaces__msg__JoystickData.
typedef struct my_interfaces__msg__JoystickData__Sequence
{
  my_interfaces__msg__JoystickData * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} my_interfaces__msg__JoystickData__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MY_INTERFACES__MSG__DETAIL__JOYSTICK_DATA__STRUCT_H_
