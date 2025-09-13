// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from my_interfaces:msg/SingleTargetEstimate.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "my_interfaces/msg/detail/single_target_estimate__rosidl_typesupport_introspection_c.h"
#include "my_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "my_interfaces/msg/detail/single_target_estimate__functions.h"
#include "my_interfaces/msg/detail/single_target_estimate__struct.h"


// Include directives for member types
// Member `position`
#include "geometry_msgs/msg/point.h"
// Member `position`
#include "geometry_msgs/msg/detail/point__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__SingleTargetEstimate_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  my_interfaces__msg__SingleTargetEstimate__init(message_memory);
}

void my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__SingleTargetEstimate_fini_function(void * message_memory)
{
  my_interfaces__msg__SingleTargetEstimate__fini(message_memory);
}

size_t my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__size_function__SingleTargetEstimate__covariance(
  const void * untyped_member)
{
  (void)untyped_member;
  return 9;
}

const void * my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__get_const_function__SingleTargetEstimate__covariance(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__get_function__SingleTargetEstimate__covariance(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__fetch_function__SingleTargetEstimate__covariance(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__get_const_function__SingleTargetEstimate__covariance(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__assign_function__SingleTargetEstimate__covariance(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__get_function__SingleTargetEstimate__covariance(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__SingleTargetEstimate_message_member_array[3] = {
  {
    "target_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__SingleTargetEstimate, target_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__SingleTargetEstimate, position),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "covariance",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    9,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__SingleTargetEstimate, covariance),  // bytes offset in struct
    NULL,  // default value
    my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__size_function__SingleTargetEstimate__covariance,  // size() function pointer
    my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__get_const_function__SingleTargetEstimate__covariance,  // get_const(index) function pointer
    my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__get_function__SingleTargetEstimate__covariance,  // get(index) function pointer
    my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__fetch_function__SingleTargetEstimate__covariance,  // fetch(index, &value) function pointer
    my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__assign_function__SingleTargetEstimate__covariance,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__SingleTargetEstimate_message_members = {
  "my_interfaces__msg",  // message namespace
  "SingleTargetEstimate",  // message name
  3,  // number of fields
  sizeof(my_interfaces__msg__SingleTargetEstimate),
  my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__SingleTargetEstimate_message_member_array,  // message members
  my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__SingleTargetEstimate_init_function,  // function to initialize message memory (memory has to be allocated)
  my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__SingleTargetEstimate_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__SingleTargetEstimate_message_type_support_handle = {
  0,
  &my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__SingleTargetEstimate_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_my_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, my_interfaces, msg, SingleTargetEstimate)() {
  my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__SingleTargetEstimate_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  if (!my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__SingleTargetEstimate_message_type_support_handle.typesupport_identifier) {
    my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__SingleTargetEstimate_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &my_interfaces__msg__SingleTargetEstimate__rosidl_typesupport_introspection_c__SingleTargetEstimate_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
